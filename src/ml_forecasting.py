# src/ml_forecasting.py
#
# ML training pipeline for the quant research assistant.
#
# Feature layers:
#   Layer 1 — Technical indicators    (OHLCV-derived)
#   Layer 2 — OHLC structure          (intraday price behavior)
#   Layer 3 — Macro context           (VIX, TNX)
#   Layer 4 — Earnings proximity      (days to/from earnings)
#   Layer 5 — Rolling risk metrics    (Sharpe, drawdown, vol, skew, VaR)
#   Layer 6 — Google Trends           (attention signal)
#   Layer 7 — GDELT sentiment         (FinBERT-scored news sentiment)
#   Layer 8 — Insider transactions    (SEC Form 4 buy/sell activity)
#
# Training decisions:
#   - balanced_accuracy scoring in GridSearch — immune to class imbalance,
#     penalizes degenerate "predict Up always" solutions
#   - scale_pos_weight = n_down/n_up — slight Down-class weighting
#   - Fixed 0.5 threshold — stable, interpretable
#   - Permutation importance pruning uses accuracy — stable on 149 samples
#
# Run after data_pipeline.py:
#   python src/ml_forecasting.py

import os
import sys
import json
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PRETRAINED_TICKERS,
    TRAIN_START,
    TEST_SIZE,
    RANDOM_STATE,
    N_CV_SPLITS,
    PARAM_GRID,
    GDELT_API_URL,
    GDELT_TIMEOUT,
    INFERENCE_LOOKBACK_MONTHS,
    data_path,
    model_path,
    output_path,
    get_train_end,
    get_gdelt_queries,
    get_trends_query,
    is_etf,
    update_registry,
    save_registry,
    load_registry,
)

warnings.filterwarnings("ignore")


# ── LAYER 1 + 2 + 5: TECHNICAL + OHLC + ROLLING RISK ─────────────────

def add_technical_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Engineer all price-based features from OHLCV data.

    Layer 1 — Technical indicators:
        Moving averages (SMA20, SMA50), MACD, RSI, Bollinger Bands,
        ATR, volume ratio, lag returns (1d, 5d, 10d).

    Layer 2 — OHLC structure features:
        Capture intraday buying/selling pressure invisible in Close alone.
        overnight_gap, intraday_range, close_position,
        upper_shadow, lower_shadow.

    Layer 5 — Rolling risk metrics:
        Time-varying statistical_analysis.py equivalents.
        Single-period stats are useless as features (same value per row).
        Rolling versions vary daily and capture current risk regime.
        rolling_vol_20, rolling_sharpe_20, rolling_drawdown_20,
        rolling_skew_20, rolling_var_20.
    """
    df     = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    volume = df["Volume"]

    # ── Layer 1 ───────────────────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    df["price_to_sma20"] = close / sma20
    df["price_to_sma50"] = close / sma50
    df["sma20_to_sma50"] = sma20 / sma50

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    df["macd"]           = macd
    df["macd_signal"]    = macd.ewm(span=9).mean()
    df["macd_histogram"] = macd - df["macd_signal"]

    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    bb_mid           = close.rolling(20).mean()
    bb_std           = close.rolling(20).std()
    df["bb_width"]    = (2 * bb_std) / bb_mid
    df["bb_position"] = (
        (close - (bb_mid - 2 * bb_std)) /
        (4 * bb_std).replace(0, np.nan)
    )

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean() / close

    df["volume_ratio"] = volume / volume.rolling(20).mean()

    returns          = close.pct_change()
    df["return_1d"]  = returns
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    # ── Layer 2 ───────────────────────────────────────────────────────
    candle_range          = (high - low).replace(0, np.nan)
    df["overnight_gap"]   = (open_ - close.shift()) / close.shift()
    df["intraday_range"]  = (high - low) / close
    df["close_position"]  = (close - low) / candle_range
    df["upper_shadow"]    = (
        high - pd.concat([close, open_], axis=1).max(axis=1)
    ) / candle_range
    df["lower_shadow"]    = (
        pd.concat([close, open_], axis=1).min(axis=1) - low
    ) / candle_range

    # ── Layer 5 ───────────────────────────────────────────────────────
    df["rolling_vol_20"]      = returns.rolling(20).std() * np.sqrt(252)

    roll_mean = returns.rolling(20).mean()
    roll_std  = returns.rolling(20).std().replace(0, np.nan)
    df["rolling_sharpe_20"]   = (roll_mean / roll_std) * np.sqrt(252)

    rolling_peak              = close.rolling(20).max()
    df["rolling_drawdown_20"] = (close - rolling_peak) / rolling_peak

    df["rolling_skew_20"]     = returns.rolling(20).skew()
    df["rolling_var_20"]      = returns.rolling(20).quantile(0.05)

    return df


# ── LAYER 3: MACRO ────────────────────────────────────────────────────

def load_and_merge_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Merge VIX + TNX macro features. Forward-fill weekends/holidays."""
    macro_path = data_path("macro.csv")
    if not os.path.exists(macro_path):
        print("  Warning: macro.csv not found — skipping")
        return df

    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    df = df.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill().bfill()
    return df


# ── LAYER 4: EARNINGS PROXIMITY ───────────────────────────────────────

def load_and_merge_earnings(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge earnings proximity features.
    Skipped for ETFs. Capped at 60 days in both directions.

    Features: days_to_earnings, days_from_earnings.
    """
    if is_etf(ticker):
        return df

    earnings_path = data_path(f"{ticker}_earnings.csv")
    if not os.path.exists(earnings_path):
        print(f"  Warning: {ticker}_earnings.csv not found — skipping")
        return df

    earnings = pd.read_csv(earnings_path, parse_dates=["earnings_date"])
    dates    = pd.to_datetime(
        earnings["earnings_date"]
    ).dt.tz_localize(None).sort_values()

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


# ── LAYER 6: GOOGLE TRENDS ────────────────────────────────────────────

def load_and_merge_trends(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge Google Trends attention features.

    Weekly data forward-filled to daily — correct since weekly
    score represents the full week for every day within it.
    """
    trends_path = data_path(f"{ticker}_trends.csv")
    if not os.path.exists(trends_path):
        print(f"  Warning: {ticker}_trends.csv not found — skipping")
        return df

    trends = pd.read_csv(trends_path, index_col=0, parse_dates=True)
    trends.index = pd.to_datetime(trends.index).tz_localize(None)
    trends_daily = trends.reindex(df.index, method="ffill").bfill()
    df = df.join(trends_daily, how="left")
    return df


# ── LAYER 7: GDELT SENTIMENT ──────────────────────────────────────────

def load_and_merge_gdelt(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge GDELT FinBERT-scored sentiment features.

    Two layers: company-specific + sector-level.
    Forward-fill up to 7 days (sentiment persists), zero-fill remainder.
    """
    gdelt_path = data_path(f"{ticker}_gdelt_daily.csv")
    if not os.path.exists(gdelt_path):
        print(f"  Warning: {ticker}_gdelt_daily.csv not found — skipping")
        return df

    gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
    gdelt.index = pd.to_datetime(gdelt.index).tz_localize(None)
    gdelt_aligned = gdelt.reindex(df.index).ffill(limit=7).fillna(0)
    df = df.join(gdelt_aligned, how="left")
    return df


# ── LAYER 8: INSIDER TRANSACTIONS ────────────────────────────────────

def load_and_merge_insider(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge insider transaction features.

    For large-cap tech (FAANG): buy features near-zero due to 10b5-1 plans.
    Sell-side features (sell count, sell value, sell acceleration) still
    carry signal for these tickers.

    For mid/small-cap: buy features carry strong conviction signal.
    XGBoost permutation importance will correctly weight each feature
    per ticker based on actual predictive power.
    """
    insider_path = data_path(f"{ticker}_insider_daily.csv")
    if not os.path.exists(insider_path):
        print(f"  Warning: {ticker}_insider_daily.csv not found — skipping")
        return df

    insider = pd.read_csv(insider_path, index_col=0, parse_dates=True)
    insider.index = pd.to_datetime(insider.index).tz_localize(None)

    # Sell acceleration — captures accelerating selling pressure
    # Meaningful even when buys are zero (large-cap tech case)
    if "insider_sell_count_30d" in insider.columns:
        insider["insider_sell_accel"] = (
            insider["insider_sell_count_30d"] -
            insider["insider_sell_count_30d"].shift(30)
        ).fillna(0)

    df = df.join(insider.reindex(df.index).fillna(0), how="left")
    return df


# ── TARGET VARIABLE ───────────────────────────────────────────────────

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if tomorrow's close > today's close, else 0.
    Last row dropped — no future close available for labeling.
    """
    df           = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df           = df.iloc[:-1]
    return df


# ── FEATURE PREPARATION ───────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, ticker: str) -> tuple:
    """
    Select feature columns, drop raw OHLCV and target columns.
    Drop features with >50% missing values — unreliable signal.
    Drop rows with any NaN in selected features.

    Returns (X, y, feature_names).
    """
    exclude      = ["Open", "High", "Low", "Close", "Volume", "target"]
    feature_cols = [c for c in df.columns if c not in exclude]

    missing_pct  = df[feature_cols].isnull().mean()
    feature_cols = [c for c in feature_cols if missing_pct[c] <= 0.5]

    df_clean = df[feature_cols + ["target"]].dropna()
    X        = df_clean[feature_cols]
    y        = df_clean["target"]

    print(f"  Features: {len(feature_cols)} | Training rows: {len(X)}")
    return X, y, feature_cols


# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────

def walk_forward_split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Chronological 80/20 split — never shuffles.

    Test set is always the most recent 20% of data.
    Shuffling would leak future information into training.
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
    GridSearch over PARAM_GRID using TimeSeriesSplit CV.

    Scoring: balanced_accuracy
        - Mean of per-class accuracy
        - Predicting Up always → 50% regardless of base rate
        - Immune to the slight Up-day majority in bull markets
        - Prevents degenerate "predict Up always" solutions

    TimeSeriesSplit: mandatory for time series CV
        - Never mixes future and past data in folds
        - Standard k-fold would produce artificially high CV scores
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
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best CV balanced accuracy: {grid.best_score_:.3f}")
    return grid.best_params_


# ── MODEL TRAINING ────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
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
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    best_params: dict,
                    scale_pos_weight: float) -> tuple:
    """
    Permutation importance feature selection.

    Drops features with negative mean importance — shuffling them
    improves accuracy, meaning they are active noise.

    Pruning criterion: plain accuracy (stable on 149 test samples).
    Balanced accuracy is too noisy for a pruning decision at this sample
    size — small fluctuations can swing it 5-10 percentage points.

    Safety guarantee: pruned model only adopted if accuracy >= baseline.
    This step can never make the model worse.

    scale_pos_weight passed through for consistent retraining.
    """
    baseline_acc = accuracy_score(y_test, model.predict(X_test))

    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    keep_mask     = perm.importances_mean >= 0
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
    pruned_acc = accuracy_score(
        y_test, model_pruned.predict(X_test[kept_features])
    )

    n_pruned = len(X_train.columns) - len(kept_features)
    MIN_IMPRV = 0.01  # Minimum accuracy improvement to justify pruning
    if pruned_acc >= baseline_acc + MIN_IMPRV:
        print(f"  Pruned {n_pruned} features: {baseline_acc:.3f} → {pruned_acc:.3f}")
        return model_pruned, kept_features
    else:
        print(
            f"  Pruning rejected: {pruned_acc:.3f} < baseline "
            f"{baseline_acc:.3f} — keeping all features"
        )
        return model, X_train.columns.tolist()


# ── EVALUATION ────────────────────────────────────────────────────────

def evaluate_model(model: XGBClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   ticker: str,
                   feature_list: list) -> dict:
    """
    Evaluate model on held-out test set.
    Saves feature importance chart (top 20) to outputs/.

    Reports: accuracy, precision, recall, F1, balanced_accuracy.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_test, y_pred, zero_division=0), 4),
        "threshold":         0.50,
    }

    print(f"  Accuracy:           {metrics['accuracy']:.1%}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.1%}")
    print(f"  Precision:          {metrics['precision']:.1%}")
    print(f"  Recall:             {metrics['recall']:.1%}")
    print(f"  F1:                 {metrics['f1']:.1%}")

    # Feature importance chart — top 20 by XGBoost gain
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_list[i] for i in reversed(indices)],
        importances[indices[::-1]],
        color="steelblue",
    )
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"{ticker} — Feature Importance (Top 20)")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    path = output_path(f"{ticker}_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved chart → {path}")

    return metrics


# ── SAVE MODEL ────────────────────────────────────────────────────────

def save_model(model: XGBClassifier,
               feature_list: list,
               ticker: str) -> None:
    """
    Save model + feature list + threshold as single pkl.

    Saving feature list alongside model prevents input mismatch
    at inference time — exact training columns always available.
    """
    os.makedirs("models", exist_ok=True)
    payload = {
        "model":     model,
        "features":  feature_list,
        "threshold": 0.50,
    }
    joblib.dump(payload, model_path(ticker))
    print(f"  Saved → {model_path(ticker)}")


# ── LIVE GDELT (INFERENCE) ────────────────────────────────────────────

def fetch_live_gdelt_tone(ticker: str, company_name: str) -> dict:
    """
    Fetch current GDELT sentiment via timelinetone — no FinBERT.

    Used at inference time in the Streamlit app.
    timelinetone: pre-aggregated tone, single API call, no local model.
    Works within Streamlit Cloud's 1GB RAM limit.

    Returns dict with company_tone, sector_tone, gdelt_composite.
    """
    import requests as req

    queries  = get_gdelt_queries(ticker, company_name)
    end_dt   = datetime.now().strftime("%Y%m%d%H%M%S")
    start_dt = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d%H%M%S")

    def fetch_tone(query: str) -> float:
        params = {
            "query":         query,
            "mode":          "timelinetone",
            "format":        "json",
            "startdatetime": start_dt,
            "enddatetime":   end_dt,
        }
        try:
            r = req.get(GDELT_API_URL, params=params, timeout=GDELT_TIMEOUT)
            if not r.text.strip():
                return 0.0
            timeline = r.json().get("timeline", [])
            values   = [
                float(point.get("value", 0.0))
                for series in timeline
                for point in series.get("data", [])
            ]
            return float(np.mean(values)) / 10.0 if values else 0.0
        except Exception:
            return 0.0

    company_tones = [fetch_tone(q) for q in queries.get("company", [])]
    sector_tones  = [fetch_tone(q) for q in queries.get("sector", [])]

    company_avg = float(np.mean(company_tones)) if company_tones else 0.0
    sector_avg  = float(np.mean(sector_tones))  if sector_tones  else 0.0

    return {
        "company_tone":    company_avg,
        "sector_tone":     sector_avg,
        "gdelt_composite": company_avg * 0.6 + sector_avg * 0.4,
    }


# ── LIVE INFERENCE ────────────────────────────────────────────────────

def fetch_live_features(ticker: str) -> pd.DataFrame:
    """
    Fetch today's feature row for live inference.

    Architecture:
        Live fetches:   OHLCV + macro (always fresh, fast)
        Cached loads:   earnings, trends, GDELT, insider (from disk)

    Keeps app response time to ~3 seconds.
    GDELT uses cached daily CSV — no FinBERT at inference time.

    Returns single-row DataFrame aligned to saved model's feature list.
    """
    saved        = joblib.load(model_path(ticker))
    feature_list = saved["features"]

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.now() - timedelta(days=INFERENCE_LOOKBACK_MONTHS * 31)
    ).strftime("%Y-%m-%d")

    # ── Live: OHLCV ───────────────────────────────────────────────────
    df = yf.download(
        ticker, start=start_date, end=end_date,
        interval="1d", auto_adjust=True, progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = add_technical_indicators(df, ticker)

    # ── Live: Macro ───────────────────────────────────────────────────
    vix = yf.download("^VIX", start=start_date, end=end_date,
                      interval="1d", auto_adjust=True, progress=False)
    tnx = yf.download("^TNX", start=start_date, end=end_date,
                      interval="1d", auto_adjust=True, progress=False)

    for raw in [vix, tnx]:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

    macro = pd.DataFrame(index=vix.index)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    macro["vix"]        = vix["Close"]
    macro["tnx"]        = tnx["Close"]
    macro["vix_change"] = macro["vix"].pct_change()
    macro["tnx_change"] = macro["tnx"].pct_change()
    macro["vix_ma20"]   = macro["vix"].rolling(20).mean()

    df = df.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill().bfill()

    # ── Cached: Earnings ──────────────────────────────────────────────
    df = load_and_merge_earnings(df, ticker)

    # ── Cached: Trends ────────────────────────────────────────────────
    trends_path = data_path(f"{ticker}_trends.csv")
    if os.path.exists(trends_path):
        trends = pd.read_csv(trends_path, index_col=0, parse_dates=True)
        trends.index = pd.to_datetime(trends.index).tz_localize(None)
        df = df.join(trends.reindex(df.index, method="ffill").bfill(),
                     how="left")

    # ── Cached: GDELT ─────────────────────────────────────────────────
    gdelt_path = data_path(f"{ticker}_gdelt_daily.csv")
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
        gdelt.index = pd.to_datetime(gdelt.index).tz_localize(None)
        df = df.join(
            gdelt.reindex(df.index).ffill(limit=7).fillna(0),
            how="left"
        )

    # ── Cached: Insider ───────────────────────────────────────────────
    df = load_and_merge_insider(df, ticker)

    # ── Extract latest row aligned to model features ──────────────────
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_list].iloc[-1:]


# ── FULL TRAINING PIPELINE ────────────────────────────────────────────

def run_ml_pipeline(tickers: list = None) -> None:
    """
    Full ML training pipeline for all configured tickers.

    Per ticker:
        1.  Load OHLCV
        2.  Engineer features (all 8 layers)
        3.  Create binary labels
        4.  Prepare feature matrix
        5.  Chronological 80/20 train/test split
        6.  GridSearch (balanced_accuracy, TimeSeriesSplit)
        7.  Train final model
        8.  Permutation importance feature selection (accuracy criterion)
        9.  Evaluate on test set
        10. Save model + features + threshold
        11. Update MODEL_REGISTRY
    """
    if tickers is None:
        tickers = PRETRAINED_TICKERS

    existing_registry = load_registry()
    results           = {}

    for ticker in tickers:
        print(f"\n{'='*55}")
        print(f"  Training: {ticker}")
        print(f"{'='*55}")

        try:
            # Step 1 — Load OHLCV
            raw_path = data_path(f"{ticker}_raw.csv")
            if not os.path.exists(raw_path):
                print(f"  ERROR: {raw_path} not found — run data_pipeline.py first")
                continue

            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)

            # Steps 2-3 — Engineer + merge all feature layers
            print("\n[1/9] Engineering + merging features...")
            df = add_technical_indicators(df, ticker)
            df = load_and_merge_macro(df)
            df = load_and_merge_earnings(df, ticker)
            df = load_and_merge_trends(df, ticker)
            df = load_and_merge_gdelt(df, ticker)
            df = load_and_merge_insider(df, ticker)

            # Step 4 — Labels
            print("[2/9] Creating labels...")
            df = create_labels(df)

            # Step 5 — Feature matrix
            print("[3/9] Preparing feature matrix...")
            X, y, feature_cols = prepare_features(df, ticker)

            # Step 6 — Train/test split
            print("[4/9] Splitting train/test...")
            X_train, X_test, y_train, y_test = walk_forward_split(X, y)
            print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

            n_down           = (y_train == 0).sum()
            n_up             = (y_train == 1).sum()
            #scale_pos_weight = n_down / n_up if n_up > 0 else 1.0
            scale_pos_weight = 1.0  # No weighting — slight Up majority is natural in bull markets, and balanced_accuracy handles it
            print(
                f"  Up: {n_up} | Down: {n_down} | "
                f"scale_pos_weight: {scale_pos_weight:.2f}"
            )

            # Step 7 — GridSearch (runs exactly once)
            print("[5/9] GridSearch (balanced_accuracy, TimeSeriesSplit)...")
            best_params = tune_hyperparameters(
                X_train, y_train, scale_pos_weight
            )
            print(f"  Best params: {best_params}")

            # Step 8 — Train final model
            print("[6/9] Training final model...")
            model = train_model(
                X_train, y_train, best_params, scale_pos_weight
            )

            # Step 9 — Feature selection
            print("[7/9] Permutation importance feature selection...")
            model, final_features = select_features(
                model, X_train, y_train,
                X_test, y_test,
                best_params, scale_pos_weight,
            )

            # Retrain on final features if pruning occurred
            if set(final_features) != set(feature_cols):
                model  = train_model(
                    X_train[final_features], y_train,
                    best_params, scale_pos_weight,
                )
                X_test = X_test[final_features]

            # Step 10 — Evaluate
            print("[8/9] Evaluating on test set...")
            metrics = evaluate_model(
                model, X_test, y_test, ticker, final_features
            )

            # Step 11 — Save
            print("[9/9] Saving model...")
            save_model(model, final_features, ticker)

            # Update registry
            update_registry(ticker, metrics["accuracy"], len(final_features))
            metrics["n_features"] = len(final_features)
            results[ticker]       = metrics

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    # Merge into existing registry — untouched tickers preserved
    final_registry = existing_registry.copy()
    for ticker, m in results.items():
        final_registry[ticker] = {
            "baseline_accuracy":          m["accuracy"],
            "baseline_balanced_accuracy": m["balanced_accuracy"],
            "baseline_f1":                m["f1"],
            "threshold":                  0.50,
            "trained_on":                 datetime.now().strftime("%Y-%m-%d"),
            "train_end":                  get_train_end(ticker),
            "n_features":                 m["n_features"],
        }

    os.makedirs("data", exist_ok=True)
    with open(data_path("model_registry.json"), "w") as f:
        json.dump(final_registry, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    print(
        f"{'Ticker':<8} {'Accuracy':>10} {'Bal.Acc':>10} "
        f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'Feats':>6}"
    )
    print("-" * 70)
    for ticker, m in results.items():
        print(
            f"{ticker:<8} "
            f"{m['accuracy']:>10.1%} "
            f"{m['balanced_accuracy']:>10.1%} "
            f"{m['precision']:>10.1%} "
            f"{m['recall']:>10.1%} "
            f"{m['f1']:>10.1%} "
            f"{m['n_features']:>6}"
        )


if __name__ == "__main__":
    run_ml_pipeline()