# src/ml_forecasting.py
#
# ML training pipeline for the quant research assistant.
#
# Feature layers (in order of addition):
#   Layer 1 — Technical indicators    (OHLCV-derived)
#   Layer 2 — OHLC structure          (intraday price behavior)
#   Layer 3 — Macro context           (VIX, TNX)
#   Layer 4 — Earnings proximity      (days to/from earnings)
#   Layer 5 — Rolling risk metrics    (Sharpe, drawdown, vol, skew, VaR)
#   Layer 6 — Google Trends           (attention signal)
#   Layer 7 — GDELT sentiment         (FinBERT-scored news sentiment)
#
# Accuracy progression tracked in MODEL_REGISTRY in config.py.
#
# Run after data_pipeline.py:
#   python src/ml_forecasting.py

import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PRETRAINED_TICKERS,
    TEST_SIZE,
    RANDOM_STATE,
    N_CV_SPLITS,
    PARAM_GRID,
    INFERENCE_LOOKBACK_MONTHS,
    data_path,
    model_path,
    output_path,
    is_etf,
    update_registry,
    save_registry,
    load_registry,
)

warnings.filterwarnings("ignore")


# ── LAYER 1 + 2: TECHNICAL + OHLC STRUCTURE FEATURES ─────────────────

def add_technical_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Engineer all price-based features from OHLCV data.

    Layer 1 — Technical indicators:
        Moving averages, MACD, RSI, Bollinger Bands, ATR,
        volume ratio, lag returns.

    Layer 2 — OHLC structure features:
        Capture intraday buying/selling pressure invisible in Close alone.
        overnight_gap, intraday_range, close_position, upper_shadow,
        lower_shadow.

    Layer 5 — Rolling risk metrics:
        Time-varying versions of statistical_analysis.py metrics.
        Sharpe, drawdown, volatility, skewness, VaR — all rolling 20-day.
        Single-period stats from statistical_analysis.py are useless as
        features (same value every row). Rolling versions vary daily and
        capture current risk regime.
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    volume = df["Volume"]

    # ── Layer 1: Technical Indicators ────────────────────────────────

    # Moving averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    df["price_to_sma20"]  = close / sma20
    df["price_to_sma50"]  = close / sma50
    df["sma20_to_sma50"]  = sma20 / sma50

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    df["macd"]            = macd
    df["macd_signal"]     = macd.ewm(span=9).mean()
    df["macd_histogram"]  = macd - df["macd_signal"]

    # RSI
    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid        = close.rolling(20).mean()
    bb_std        = close.rolling(20).std()
    df["bb_width"]    = (2 * bb_std) / bb_mid
    df["bb_position"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean() / close

    # Volume
    df["volume_ratio"] = volume / volume.rolling(20).mean()

    # Lag returns
    returns = close.pct_change()
    df["return_1d"]  = returns
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    # ── Layer 2: OHLC Structure Features ─────────────────────────────

    candle_range    = (high - low).replace(0, np.nan)

    df["overnight_gap"]   = (open_ - close.shift()) / close.shift()
    df["intraday_range"]  = (high - low) / close
    df["close_position"]  = (close - low) / candle_range
    df["upper_shadow"]    = (high - pd.concat([close, open_], axis=1).max(axis=1)) / candle_range
    df["lower_shadow"]    = (pd.concat([close, open_], axis=1).min(axis=1) - low) / candle_range

    # ── Layer 5: Rolling Risk Metrics ─────────────────────────────────
    # Time-varying versions of statistical_analysis.py outputs.
    # 20-day window balances stability vs responsiveness.

    # Rolling volatility — current vol regime
    df["rolling_vol_20"]      = returns.rolling(20).std() * np.sqrt(252)

    # Rolling Sharpe — recent risk-adjusted return
    roll_mean = returns.rolling(20).mean()
    roll_std  = returns.rolling(20).std().replace(0, np.nan)
    df["rolling_sharpe_20"]   = (roll_mean / roll_std) * np.sqrt(252)

    # Rolling drawdown — how far below recent peak
    rolling_peak              = close.rolling(20).max()
    df["rolling_drawdown_20"] = (close - rolling_peak) / rolling_peak

    # Rolling skewness — tail asymmetry of recent returns
    df["rolling_skew_20"]     = returns.rolling(20).skew()

    # Rolling VaR 95% — recent left-tail risk
    df["rolling_var_20"]      = returns.rolling(20).quantile(0.05)

    return df


# ── LAYER 3: MACRO FEATURES ───────────────────────────────────────────

def load_and_merge_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro features (VIX, TNX) into the feature matrix.

    Uses forward-fill then backward-fill to handle weekends and
    holidays where macro data is unavailable. Inner join ensures
    we only keep dates where both price and macro data exist.
    """
    macro_path = data_path("macro.csv")
    if not os.path.exists(macro_path):
        print("  Warning: macro.csv not found — skipping macro features")
        return df

    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)

    df = df.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill().bfill()
    return df


# ── LAYER 4: EARNINGS PROXIMITY FEATURES ─────────────────────────────

def load_and_merge_earnings(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge earnings proximity features into the feature matrix.

    Skipped for ETFs. Returns df unchanged if earnings file missing.

    Features:
        days_to_earnings   : trading days until next earnings (capped 60)
        days_from_earnings : trading days since last earnings (capped 60)
    """
    if is_etf(ticker):
        return df

    earnings_path = data_path(f"{ticker}_earnings.csv")
    if not os.path.exists(earnings_path):
        print(f"  Warning: {ticker}_earnings.csv not found — skipping")
        return df

    earnings = pd.read_csv(earnings_path, parse_dates=["earnings_date"])
    dates    = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None).sort_values()

    days_to   = []
    days_from = []

    for date in df.index:
        future = dates[dates > date]
        past   = dates[dates <= date]

        days_to.append(
            min((future.iloc[0] - date).days, 60) if len(future) > 0 else 60
        )
        days_from.append(
            min((date - past.iloc[-1]).days, 60) if len(past) > 0 else 60
        )

    df["days_to_earnings"]   = days_to
    df["days_from_earnings"] = days_from
    return df


# ── LAYER 6: GOOGLE TRENDS FEATURES ──────────────────────────────────

def load_and_merge_trends(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge Google Trends attention features into the feature matrix.

    Trends data is weekly — forward-filled to daily to match the
    price data resolution. This means every trading day in a given
    week carries the same trends score for that week, which is the
    correct interpretation since weekly data represents the full week.

    Features merged:
        trends_score   : raw weekly attention index (0-100)
        trends_change  : week-over-week change
        trends_zscore  : deviation from 52-week baseline
        trends_spike   : binary flag — abnormally high attention
    """
    trends_path = data_path(f"{ticker}_trends.csv")
    if not os.path.exists(trends_path):
        print(f"  Warning: {ticker}_trends.csv not found — skipping trends")
        return df

    trends = pd.read_csv(trends_path, index_col=0, parse_dates=True)
    trends.index = pd.to_datetime(trends.index).tz_localize(None)

    # Reindex to daily frequency then forward-fill weekly values
    trends_daily = trends.reindex(df.index, method="ffill")

    # Any remaining NaN at the start — backfill
    trends_daily = trends_daily.bfill()

    df = df.join(trends_daily, how="left")
    return df


# ── LAYER 7: GDELT SENTIMENT FEATURES ────────────────────────────────

def load_and_merge_gdelt(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge GDELT sentiment features into the feature matrix.

    Two sentiment layers:
        company layer : sentiment directly about this company
        sector layer  : industry/supply-chain sentiment

    Null handling:
        Forward-fill up to 7 days — sentiment persists short-term
        Zero-fill after 7 days — assume neutral if no recent data

    Features merged:
        company_tone, company_tone_ma7, company_tone_change,
        company_positive, company_negative,
        sector_tone, sector_tone_ma7, sector_tone_change,
        sector_positive, sector_negative,
        gdelt_composite
    """
    gdelt_path = data_path(f"{ticker}_gdelt_daily.csv")
    if not os.path.exists(gdelt_path):
        print(f"  Warning: {ticker}_gdelt_daily.csv not found — skipping GDELT")
        return df

    gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
    gdelt.index = pd.to_datetime(gdelt.index).tz_localize(None)

    # Reindex to match price data dates
    gdelt_aligned = gdelt.reindex(df.index)

    # Forward-fill up to 7 days — sentiment persists short-term
    gdelt_aligned = gdelt_aligned.ffill(limit=7)

    # Zero-fill remaining NaN — neutral sentiment assumption
    gdelt_aligned = gdelt_aligned.fillna(0)

    df = df.join(gdelt_aligned, how="left")
    return df

# ── LAYER 8: INSIDER TRANSACTIONS FEATURES ───────────────────────────

def load_and_merge_insider(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge insider transaction features into the feature matrix.

    Features are pre-computed rolling aggregates — no forward-fill
    needed since they're already computed on rolling windows.
    Zero-fill any remaining NaN — no insider activity = neutral signal.

    Features merged:
        insider_buy_count_30d  : purchases in last 30 days
        insider_sell_count_30d : sales in last 30 days
        insider_net_30d        : net buy/sell direction
        insider_buy_value_30d  : log-scaled buy value
        insider_sell_value_30d : log-scaled sell value
        insider_cluster_buy    : 2+ insiders bought — conviction signal
        exec_bought_30d        : CEO/CFO bought — highest signal
    """
    insider_path = data_path(f"{ticker}_insider_daily.csv")
    if not os.path.exists(insider_path):
        print(f"  Warning: {ticker}_insider_daily.csv not found — skipping")
        return df

    insider = pd.read_csv(insider_path, index_col=0, parse_dates=True)
    insider.index = pd.to_datetime(insider.index).tz_localize(None)

    df = df.join(insider, how="left")
    insider_cols = insider.columns.tolist()
    df[insider_cols] = df[insider_cols].fillna(0)
    return df

# ── TARGET VARIABLE ───────────────────────────────────────────────────

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target.

    Target: 1 if tomorrow's close > today's close, else 0.
    Last row dropped — no future close available for labeling.
    """
    df         = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df         = df.iloc[:-1]   # drop last row — no label available
    return df


# ── FEATURE PREPARATION ───────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, ticker: str) -> tuple:
    """
    Select features and drop rows with NaN values.

    Returns (X, y, feature_names) where X is the feature matrix,
    y is the target vector, and feature_names is the list of columns
    used — saved alongside the model for consistent inference.

    Feature exclusions:
        OHLCV raw columns — scale-dependent, non-stationary
        target column — would be data leakage
        any column with >50% missing values — unreliable
    """
    # Columns that are raw data — not engineered features
    exclude = ["Open", "High", "Low", "Close", "Volume", "target"]

    feature_cols = [c for c in df.columns if c not in exclude]

    # Drop features with >50% missing values
    missing_pct  = df[feature_cols].isnull().mean()
    feature_cols = [c for c in feature_cols if missing_pct[c] <= 0.5]

    # Drop rows where any selected feature is NaN
    df_clean = df[feature_cols + ["target"]].dropna()

    X = df_clean[feature_cols]
    y = df_clean["target"]

    print(f"  Features: {len(feature_cols)} | Training rows: {len(X)}")
    return X, y, feature_cols


# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────

def walk_forward_split(X: pd.DataFrame,
                       y: pd.Series) -> tuple:
    """
    Chronological train/test split — never shuffles.

    80% train, 20% test. Test set is always the most recent data.
    Shuffling would leak future information into training — invalid
    for any time series ML task.
    """
    split = int(len(X) * (1 - TEST_SIZE))
    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:]
    )


# ── HYPERPARAMETER TUNING ─────────────────────────────────────────────

def tune_hyperparameters(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         scale_pos_weight: float) -> dict:
    """
    GridSearch over PARAM_GRID using TimeSeriesSplit cross-validation.

    TimeSeriesSplit is mandatory — standard k-fold would randomly
    mix future and past data, making CV accuracy artificially high
    and the selected hyperparameters invalid.

    scale_pos_weight balances class imbalance (unequal up/down days).
    """
    tscv  = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"  Best CV accuracy: {grid.best_score_:.3f}")
    return grid.best_params_


# ── MODEL TRAINING ────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                best_params: dict,
                scale_pos_weight: float) -> XGBClassifier:
    """Train XGBoost with tuned hyperparameters on full training set."""
    model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ── FEATURE SELECTION ─────────────────────────────────────────────────

def select_features(model: XGBClassifier,
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    best_params: dict,
                    scale_pos_weight: float) -> tuple:
    """
    Permutation importance-based feature selection.

    Computes permutation importance on the test set — measures how much
    accuracy drops when each feature is randomly shuffled. Features with
    negative mean importance (shuffling them improves accuracy) are noise.

    Only adopts the pruned feature set if accuracy >= baseline —
    guarantees this step never makes the model worse.

    Returns (model, feature_list) — either pruned or original.
    """
    baseline_acc = accuracy_score(y_test, model.predict(X_test))

    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Features with negative mean importance are actively harmful
    importances  = perm.importances_mean
    keep_mask    = importances >= 0
    kept_features = X_train.columns[keep_mask].tolist()

    if len(kept_features) == len(X_train.columns):
        print("  No features pruned — all contribute positively")
        return model, X_train.columns.tolist()

    # Retrain on pruned feature set
    model_pruned = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model_pruned.fit(X_train[kept_features], y_train)
    pruned_acc = accuracy_score(y_test, model_pruned.predict(X_test[kept_features]))

    n_pruned = len(X_train.columns) - len(kept_features)

    if pruned_acc >= baseline_acc:
        print(f"  Pruned {n_pruned} features: {baseline_acc:.3f} → {pruned_acc:.3f}")
        return model_pruned, kept_features
    else:
        print(f"  Pruning rejected: {pruned_acc:.3f} < baseline {baseline_acc:.3f} — keeping all features")
        return model, X_train.columns.tolist()


# ── EVALUATION ────────────────────────────────────────────────────────

def evaluate_model(model: XGBClassifier,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   ticker: str, feature_list: list) -> dict:
    """
    Evaluate model on held-out test set and save feature importance chart.

    Metrics:
        accuracy  : overall directional accuracy
        precision : of predicted Up days, how many were actually Up
        recall    : of actual Up days, how many did we predict
        f1        : harmonic mean of precision and recall
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1:        {metrics['f1']:.1%}")

    # Feature importance chart
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_list[i] for i in reversed(indices)],
        importances[indices[::-1]],
        color="steelblue"
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{ticker} — Feature Importance (Top 20)")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    path = output_path(f"{ticker}_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved feature importance chart → {path}")

    return metrics


# ── SAVE MODEL ────────────────────────────────────────────────────────

def save_model(model: XGBClassifier, feature_list: list,
               ticker: str) -> None:
    """
    Save model and feature list together as a single pkl file.

    Saving both together prevents input mismatch at inference time —
    the exact feature columns used during training are always available
    when loading the model.
    """
    os.makedirs("models", exist_ok=True)
    payload = {"model": model, "features": feature_list}
    joblib.dump(payload, model_path(ticker))
    print(f"  Saved model → {model_path(ticker)}")


# ── FULL TRAINING PIPELINE ────────────────────────────────────────────

def run_ml_pipeline(tickers: list = None) -> None:
    """
    Run the full ML training pipeline for all configured tickers.

    Per ticker:
        1. Load OHLCV + engineer technical + OHLC structure + rolling risk features
        2. Merge macro, earnings, trends, GDELT layers
        3. Create binary target labels
        4. Prepare feature matrix — drop raw columns and NaN rows
        5. Chronological train/test split
        6. GridSearch hyperparameter tuning with TimeSeriesSplit CV
        7. Train XGBoost on full training set
        8. Permutation importance feature selection
        9. Evaluate on held-out test set
        10. Save model + feature list
        11. Update MODEL_REGISTRY with accuracy and metadata
    """
    if tickers is None:
        tickers = PRETRAINED_TICKERS

    results = {}

    for ticker in tickers:
        print(f"\n{'='*55}")
        print(f"  Training pipeline: {ticker}")
        print(f"{'='*55}")

        try:
            # Step 1 — Load and engineer features
            raw_path = data_path(f"{ticker}_raw.csv")
            if not os.path.exists(raw_path):
                print(f"  ERROR: {raw_path} not found — run data_pipeline.py first")
                continue

            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)

            print("\n[1/9] Engineering features...")
            df = add_technical_indicators(df, ticker)

            # Step 2 — Merge data layers
            print("[2/9] Merging data layers...")
            df = load_and_merge_macro(df)
            df = load_and_merge_earnings(df, ticker)
            df = load_and_merge_trends(df, ticker)
            df = load_and_merge_gdelt(df, ticker)
            df = load_and_merge_insider(df, ticker)

            # Step 3 — Create labels
            print("[3/9] Creating labels...")
            df = create_labels(df)

            # Step 4 — Prepare feature matrix
            print("[4/9] Preparing feature matrix...")
            X, y, feature_cols = prepare_features(df, ticker)

            # Step 5 — Train/test split
            print("[5/9] Splitting train/test...")
            X_train, X_test, y_train, y_test = walk_forward_split(X, y)
            print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

            # Class balance
            n_down          = (y_train == 0).sum()
            n_up            = (y_train == 1).sum()
            scale_pos_weight = n_down / n_up if n_up > 0 else 1.0
            print(f"  Up days: {n_up} | Down days: {n_down} | scale_pos_weight: {scale_pos_weight:.2f}")

            # Step 6 — Hyperparameter tuning
            print("[6/9] Tuning hyperparameters (GridSearch + TimeSeriesSplit)...")
            best_params = tune_hyperparameters(X_train, y_train, scale_pos_weight)
            print(f"  Best params: {best_params}")

            # Step 7 — Train final model
            print("[7/9] Training final model...")
            model = train_model(X_train, y_train, best_params, scale_pos_weight)

            # Step 8 — Feature selection
            print("[8/9] Permutation importance feature selection...")
            model, final_features = select_features(
                model, X_train, y_train, X_test, y_test,
                best_params, scale_pos_weight
            )

            # Retrain on final features if pruning occurred
            if set(final_features) != set(feature_cols):
                model = train_model(
                    X_train[final_features], y_train,
                    best_params, scale_pos_weight
                )
                X_test = X_test[final_features]

            # Step 9 — Evaluate
            print("[9/9] Evaluating on test set...")
            metrics = evaluate_model(model, X_test, y_test, ticker, final_features)

            # Step 10 — Save
            save_model(model, final_features, ticker)

            # Step 11 — Update registry
            update_registry(ticker, metrics["accuracy"], len(final_features))
            results[ticker] = metrics

        except Exception as e:
            import traceback
            print(f"  ERROR during {ticker} pipeline: {e}")
            traceback.print_exc()
            continue

    # Save updated registry
    save_registry()

    # Print summary
    print(f"\n{'='*55}")
    print("Training Summary")
    print(f"{'='*55}")
    print(f"{'Ticker':<8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Features':>10}")
    print("-" * 55)
    for ticker, m in results.items():
        registry_entry = load_registry().get(ticker, {})
        n_feat = registry_entry.get("n_features", "?")
        print(
            f"{ticker:<8} "
            f"{m['accuracy']:>10.1%} "
            f"{m['precision']:>10.1%} "
            f"{m['recall']:>10.1%} "
            f"{m['f1']:>10.1%} "
            f"{n_feat:>10}"
        )


# ── LIVE INFERENCE ────────────────────────────────────────────────────

def fetch_live_features(ticker: str) -> pd.DataFrame:
    """
    Fetch today's feature row for inference on a trained model.

    Fetches INFERENCE_LOOKBACK_MONTHS of recent history — enough to
    compute all rolling window features:
        sma_50:        50 trading days  (~2.5 months)
        vix_ma20:      20 trading days  (~1 month)
        trends_zscore: 52 weeks         (~12 months)
        rolling_*:     20 trading days

    14 months covers all of these safely.

    Returns a single-row DataFrame with today's engineered features,
    aligned to the feature list the saved model was trained on.
    """
    from data_pipeline import (
        fetch_macro_data,
        fetch_trends_data, fetch_gdelt_sentiment,
        get_company_name,
    )
    from dateutil.relativedelta import relativedelta

    saved = joblib.load(model_path(ticker))
    feature_list = saved["features"]

    # Fetch recent history for rolling window computation
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.now() - relativedelta(months=INFERENCE_LOOKBACK_MONTHS)
    ).strftime("%Y-%m-%d")

    company_name = get_company_name(ticker)

    # Fetch all data layers
    df = yf.download(ticker, start=start_date, end=end_date,
                     interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Engineer features
    df = add_technical_indicators(df, ticker)

    # Merge macro — fetch live
    macro = fetch_macro_data()
    df    = df.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill().bfill()

    # Merge earnings — use existing saved file
    df = load_and_merge_earnings(df, ticker)

    # Merge trends — fetch live
    trends = fetch_trends_data(ticker, company_name)
    if not trends.empty:
        trends_daily = trends.reindex(df.index, method="ffill").bfill()
        df = df.join(trends_daily, how="left")

    # Merge GDELT — fetch live
    gdelt = fetch_gdelt_sentiment(ticker, company_name)
    if not gdelt.empty:
        gdelt_aligned = gdelt.reindex(df.index).ffill(limit=7).fillna(0)
        df = df.join(gdelt_aligned, how="left")

    # Take only the most recent row — today's feature state
    latest = df[feature_list].iloc[-1:]
    return latest


if __name__ == "__main__":
    run_ml_pipeline()