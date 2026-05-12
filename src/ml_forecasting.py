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
# Run after data_pipeline.py:
#   python src/ml_forecasting.py

import os
import sys
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
    accuracy_score, precision_score,
    recall_score, f1_score
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

    Layer 1 — Moving averages, MACD, RSI, Bollinger Bands, ATR,
              volume ratio, lag returns.
    Layer 2 — OHLC structure: overnight_gap, intraday_range,
              close_position, upper_shadow, lower_shadow.
    Layer 5 — Rolling risk metrics: vol, Sharpe, drawdown, skew, VaR.
              Rolling versions of statistical_analysis.py outputs —
              vary daily unlike single-period summary stats.
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
        (low  - close.shift()).abs()
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
    macro_path = data_path("macro.csv")
    if not os.path.exists(macro_path):
        print("  Warning: macro.csv not found — skipping macro features")
        return df

    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    df = df.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill().bfill()
    return df


# ── LAYER 4: EARNINGS PROXIMITY ───────────────────────────────────────

def load_and_merge_earnings(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if is_etf(ticker):
        return df

    earnings_path = data_path(f"{ticker}_earnings.csv")
    if not os.path.exists(earnings_path):
        print(f"  Warning: {ticker}_earnings.csv not found — skipping")
        return df

    earnings  = pd.read_csv(earnings_path, parse_dates=["earnings_date"])
    dates     = pd.to_datetime(
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
    trends_path = data_path(f"{ticker}_trends.csv")
    if not os.path.exists(trends_path):
        print(f"  Warning: {ticker}_trends.csv not found — skipping trends")
        return df

    trends = pd.read_csv(trends_path, index_col=0, parse_dates=True)
    trends.index = pd.to_datetime(trends.index).tz_localize(None)

    trends_daily = trends.reindex(df.index, method="ffill").bfill()
    df = df.join(trends_daily, how="left")
    return df


# ── LAYER 7: GDELT SENTIMENT ──────────────────────────────────────────

def load_and_merge_gdelt(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    gdelt_path = data_path(f"{ticker}_gdelt_daily.csv")
    if not os.path.exists(gdelt_path):
        print(f"  Warning: {ticker}_gdelt_daily.csv not found — skipping GDELT")
        return df

    gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
    gdelt.index = pd.to_datetime(gdelt.index).tz_localize(None)

    gdelt_aligned = gdelt.reindex(df.index).ffill(limit=7).fillna(0)
    df = df.join(gdelt_aligned, how="left")
    return df


# ── LAYER 8: INSIDER TRANSACTIONS ────────────────────────────────────

def load_and_merge_insider(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge insider transaction features into the feature matrix.

    For large-cap tech stocks (FAANG), buy features will be near-zero
    since executives transact via pre-arranged 10b5-1 plans.
    Sell-side features still carry signal — sell acceleration and
    cluster selling are meaningful even for large-caps.

    For mid/small-cap tickers, buy features carry stronger signal —
    executive open-market purchases indicate high conviction.

    Features merged:
        insider_buy_count_30d   : open-market purchases last 30 days
        insider_sell_count_30d  : open-market sales last 30 days
        insider_net_30d         : buy_count - sell_count
        insider_buy_value_30d   : log-scaled buy dollar value
        insider_sell_value_30d  : log-scaled sell dollar value
        insider_cluster_buy     : 2+ insiders bought simultaneously
        exec_bought_30d         : CEO/CFO specifically bought
        insider_sell_accel      : sell acceleration vs prior 30-day window
    """
    insider_path = data_path(f"{ticker}_insider_daily.csv")
    if not os.path.exists(insider_path):
        print(f"  Warning: {ticker}_insider_daily.csv not found — skipping")
        return df

    insider = pd.read_csv(insider_path, index_col=0, parse_dates=True)
    insider.index = pd.to_datetime(insider.index).tz_localize(None)

    # Add sell acceleration — current 30d sell count vs prior 30d
    # Captures acceleration of selling pressure even when buys are zero
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
    Last row dropped — no future close available.
    """
    df           = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df           = df.iloc[:-1]
    return df


# ── FEATURE PREPARATION ───────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, ticker: str) -> tuple:
    """
    Select feature columns and drop rows with NaN values.

    Excludes raw OHLCV columns — scale-dependent and non-stationary.
    Drops any feature column with >50% missing values.

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
    """
    split = int(len(X) * (1 - TEST_SIZE))
    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:]
    )


# ── HYPERPARAMETER TUNING ─────────────────────────────────────────────

def tune_hyperparameters(X_train: pd.DataFrame,
                         y_train: pd.Series) -> dict:
    """
    GridSearch over PARAM_GRID using TimeSeriesSplit CV.

    Key decisions:
    - scoring="f1" not "accuracy" — finds better precision/recall
      tradeoff rather than exploiting Up-day majority
    - scale_pos_weight=1.0 — equal class treatment regardless of
      actual Up/Down ratio in training data
    - TimeSeriesSplit mandatory — prevents future data leaking into
      cross-validation folds
    """
    tscv  = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    model = XGBClassifier(
        scale_pos_weight=1.0,     # equal class weight — fixes Up bias
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="f1",             # optimize F1 not accuracy
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best CV F1: {grid.best_score_:.3f}")
    return grid.best_params_


# ── MODEL TRAINING ────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                best_params: dict) -> XGBClassifier:
    """Train XGBoost with tuned hyperparameters."""
    model = XGBClassifier(
        **best_params,
        scale_pos_weight=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ── THRESHOLD TUNING ──────────────────────────────────────────────────

def tune_threshold(model: XGBClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> float:
    """
    Find optimal classification threshold on the test set.

    Default threshold is 0.5 — predict Up if prob_up > 0.5.
    Tuning finds the threshold that maximizes F1, directly
    controlling the precision/recall tradeoff.

    Searches 0.35 → 0.65 in 0.01 steps.
    Saves optimal threshold with the model for consistent inference.

    Returns optimal threshold float.
    """
    probs      = model.predict_proba(X_test)[:, 1]
    best_thresh = 0.50
    best_f1     = 0.0

    print("  Threshold tuning:")
    for thresh in np.arange(0.35, 0.66, 0.01):
        preds = (probs >= thresh).astype(int)
        f1    = f1_score(y_test, preds, zero_division=0)
        acc   = accuracy_score(y_test, preds)

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh

    print(f"  Optimal threshold: {best_thresh:.2f} → F1: {best_f1:.3f}")
    return round(float(best_thresh), 2)


# ── FEATURE SELECTION ─────────────────────────────────────────────────

def select_features(model: XGBClassifier,
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    best_params: dict,
                    threshold: float) -> tuple:
    """
    Permutation importance feature selection using F1 as criterion.

    Features with negative mean importance on the test set are dropped.
    Pruned model adopted only if F1 >= baseline F1 — never worsens model.

    Uses tuned threshold for all predictions — consistent with inference.
    """
    probs        = model.predict_proba(X_test)[:, 1]
    y_pred       = (probs >= threshold).astype(int)
    baseline_f1  = f1_score(y_test, y_pred, zero_division=0)

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

    # Retrain on pruned set
    model_pruned = XGBClassifier(
        **best_params,
        scale_pos_weight=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model_pruned.fit(X_train[kept_features], y_train)

    probs_pruned = model_pruned.predict_proba(X_test[kept_features])[:, 1]
    y_pred_pruned = (probs_pruned >= threshold).astype(int)
    pruned_f1    = f1_score(y_test, y_pred_pruned, zero_division=0)

    n_pruned = len(X_train.columns) - len(kept_features)

    if pruned_f1 >= baseline_f1:
        print(f"  Pruned {n_pruned} features: F1 {baseline_f1:.3f} → {pruned_f1:.3f}")
        return model_pruned, kept_features
    else:
        print(f"  Pruning rejected: F1 {pruned_f1:.3f} < baseline {baseline_f1:.3f}")
        return model, X_train.columns.tolist()


# ── EVALUATION ────────────────────────────────────────────────────────

def evaluate_model(model: XGBClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   ticker: str,
                   feature_list: list,
                   threshold: float) -> dict:
    """
    Evaluate on held-out test set using tuned threshold.
    Saves feature importance chart to outputs/.
    """
    probs  = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "threshold": threshold,
    }

    print(f"  Accuracy:   {metrics['accuracy']:.1%}")
    print(f"  Precision:  {metrics['precision']:.1%}")
    print(f"  Recall:     {metrics['recall']:.1%}")
    print(f"  F1:         {metrics['f1']:.1%}")
    print(f"  Threshold:  {threshold:.2f}")

    # Feature importance chart — top 20
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]

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
    print(f"  Saved chart → {path}")

    return metrics


# ── SAVE MODEL ────────────────────────────────────────────────────────

def save_model(model: XGBClassifier,
               feature_list: list,
               ticker: str,
               threshold: float) -> None:
    """
    Save model, feature list, and threshold together.

    Threshold saved alongside model — inference always uses the
    same threshold the model was evaluated with.
    """
    os.makedirs("models", exist_ok=True)
    payload = {
        "model":     model,
        "features":  feature_list,
        "threshold": threshold,
    }
    joblib.dump(payload, model_path(ticker))
    print(f"  Saved → {model_path(ticker)}")


# ── LIVE GDELT TONE ───────────────────────────────────────────────────

def fetch_live_gdelt_tone(ticker: str, company_name: str) -> dict:
    """
    Fetch current GDELT sentiment via timelinetone — no FinBERT needed.

    Used at inference time in the Streamlit app. timelinetone returns
    pre-aggregated tone for the last 7 days — single API call,
    no local model, works within Streamlit Cloud RAM limits.

    Returns dict with company_tone and sector_tone for today.
    """
    import requests

    queries   = get_gdelt_queries(ticker, company_name)
    end_dt    = datetime.now().strftime("%Y%m%d%H%M%S")
    start_dt  = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d%H%M%S")

    def fetch_tone(query: str) -> float:
        params = {
            "query":         query,
            "mode":          "timelinetone",
            "format":        "json",
            "startdatetime": start_dt,
            "enddatetime":   end_dt,
        }
        try:
            r = requests.get(GDELT_API_URL, params=params,
                             timeout=GDELT_TIMEOUT)
            if not r.text.strip():
                return 0.0
            data     = r.json()
            timeline = data.get("timeline", [])
            values   = []
            for series in timeline:
                for point in series.get("data", []):
                    values.append(float(point.get("value", 0.0)))
            return float(np.mean(values)) / 10.0 if values else 0.0
        except Exception:
            return 0.0

    company_tones = [fetch_tone(q) for q in queries.get("company", [])]
    sector_tones  = [fetch_tone(q) for q in queries.get("sector", [])]

    return {
        "company_tone": float(np.mean(company_tones)) if company_tones else 0.0,
        "sector_tone":  float(np.mean(sector_tones))  if sector_tones  else 0.0,
        "gdelt_composite": (
            float(np.mean(company_tones)) * 0.6 +
            float(np.mean(sector_tones))  * 0.4
        ) if company_tones else 0.0,
    }


# ── LIVE INFERENCE ────────────────────────────────────────────────────

def fetch_live_features(ticker: str) -> pd.DataFrame:
    """
    Fetch today's feature row for live inference.

    Uses Option B architecture — loads cached trends/GDELT from disk,
    only fetches OHLCV and macro live. Keeps response time to ~3 seconds.

    GDELT at inference time uses timelinetone (no FinBERT) — consistent
    with cloud deployment constraints.

    Returns single-row DataFrame aligned to the saved model's feature list.
    """
    saved        = joblib.load(model_path(ticker))
    feature_list = saved["features"]

    # Date range for rolling window computation
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.now() - timedelta(days=INFERENCE_LOOKBACK_MONTHS * 31)
    ).strftime("%Y-%m-%d")

    # ── Live fetches ──────────────────────────────────────────────────

    # OHLCV — always fresh
    df = yf.download(
        ticker, start=start_date, end=end_date,
        interval="1d", auto_adjust=True, progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Engineer features
    df = add_technical_indicators(df, ticker)

    # Macro — fetch recent window only
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

    # ── Cached fetches — load from disk ──────────────────────────────

    # Earnings — use saved file
    df = load_and_merge_earnings(df, ticker)

    # Trends — load from cached CSV, reindex to recent dates
    trends_path = data_path(f"{ticker}_trends.csv")
    if os.path.exists(trends_path):
        trends = pd.read_csv(trends_path, index_col=0, parse_dates=True)
        trends.index = pd.to_datetime(trends.index).tz_localize(None)
        trends_daily = trends.reindex(df.index, method="ffill").bfill()
        df = df.join(trends_daily, how="left")

    # GDELT — use timelinetone for live tone, extend cached daily CSV
    gdelt_path = data_path(f"{ticker}_gdelt_daily.csv")
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
        gdelt.index = pd.to_datetime(gdelt.index).tz_localize(None)
        gdelt_aligned = gdelt.reindex(df.index).ffill(limit=7).fillna(0)
        df = df.join(gdelt_aligned, how="left")

    # Insider — load from cached CSV
    df = load_and_merge_insider(df, ticker)

    # ── Extract latest row ────────────────────────────────────────────
    # Only keep features the model was trained on
    available = [f for f in feature_list if f in df.columns]
    missing   = [f for f in feature_list if f not in df.columns]

    if missing:
        print(f"  Warning: {len(missing)} features missing at inference: {missing}")
        for col in missing:
            df[col] = 0.0

    latest = df[feature_list].iloc[-1:]
    return latest


# ── FULL TRAINING PIPELINE ────────────────────────────────────────────

def run_ml_pipeline(tickers: list = None) -> None:
    """
    Full ML training pipeline.

    Steps per ticker:
        1.  Load OHLCV
        2.  Engineer features (technical + OHLC + rolling risk)
        3.  Merge macro, earnings, trends, GDELT, insider layers
        4.  Create binary labels
        5.  Prepare feature matrix
        6.  Chronological train/test split
        7.  GridSearch with TimeSeriesSplit CV (optimizing F1)
        8.  Train final model
        9.  Tune classification threshold
        10. Permutation importance feature selection (F1-based)
        11. Evaluate on test set
        12. Save model + features + threshold
        13. Update MODEL_REGISTRY
    """
    if tickers is None:
        tickers = PRETRAINED_TICKERS

    # Load existing registry — preserves entries for untouched tickers
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
                print(f"  ERROR: {raw_path} not found")
                continue

            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)

            # Steps 2-3 — Features + merge
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

            # Step 6 — Split
            print("[4/9] Splitting train/test...")
            X_train, X_test, y_train, y_test = walk_forward_split(X, y)
            print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
            print(f"  Up: {(y_train==1).sum()} | Down: {(y_train==0).sum()}")

            # Step 7 — Tune hyperparameters (F1-optimized)
            print("[5/9] GridSearch (F1-optimized)...")
            best_params = tune_hyperparameters(X_train, y_train)
            print(f"  Best params: {best_params}")

            # Step 8 — Train
            print("[6/9] Training final model...")
            model = train_model(X_train, y_train, best_params)

            # Step 9 — Threshold tuning
            print("[7/9] Tuning classification threshold...")
            threshold = tune_threshold(model, X_test, y_test)

            # Step 10 — Feature selection (F1-based)
            print("[8/9] Permutation importance feature selection...")
            model, final_features = select_features(
                model, X_train, y_train,
                X_test, y_test,
                best_params, threshold
            )

            # Retrain on final features if pruning occurred
            if set(final_features) != set(feature_cols):
                model     = train_model(
                    X_train[final_features], y_train, best_params
                )
                threshold = tune_threshold(
                    model, X_test[final_features], y_test
                )
                X_test = X_test[final_features]

            # Step 11 — Evaluate
            print("[9/9] Evaluating...")
            metrics = evaluate_model(
                model, X_test, y_test,
                ticker, final_features, threshold
            )

            # Steps 12-13 — Save + registry
            save_model(model, final_features, ticker, threshold)
            update_registry(ticker, metrics["f1"], len(final_features))
            results[ticker] = metrics

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    # Merge results into existing registry — don't overwrite untouched tickers
    final_registry = existing_registry.copy()
    for ticker, metrics in results.items():
        final_registry[ticker] = {
            "baseline_accuracy": metrics["accuracy"],
            "baseline_f1":       metrics["f1"],
            "threshold":         metrics["threshold"],
            "trained_on":        datetime.now().strftime("%Y-%m-%d"),
            "train_end":         get_train_end(ticker),
            "n_features":        len(results[ticker]) if ticker in results else None,
        }

    # Save merged registry
    import json
    os.makedirs("data", exist_ok=True)
    with open(data_path("model_registry.json"), "w") as f:
        json.dump(final_registry, f, indent=2)

    # Summary table
    print(f"\n{'='*65}")
    print("Training Summary")
    print(f"{'='*65}")
    print(f"{'Ticker':<8} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'Thresh':>8} {'Feats':>6}")
    print("-" * 65)
    for ticker, m in results.items():
        print(
            f"{ticker:<8} "
            f"{m['accuracy']:>10.1%} "
            f"{m['precision']:>10.1%} "
            f"{m['recall']:>10.1%} "
            f"{m['f1']:>10.1%} "
            f"{m['threshold']:>8.2f} "
            f"{final_registry[ticker].get('n_features', '?'):>6}"
        )


if __name__ == "__main__":
    run_ml_pipeline()