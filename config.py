# config.py
#
# Central configuration for the quant research assistant.
# This is the single source of truth for all settings across the project.
#
# To add a new pre-trained ticker:
#   1. Add to PRETRAINED_TICKERS
#   2. Add to TRAIN_END if cutoff differs from default
#   3. Add custom query strings to get_trends_query / get_gdelt_query (optional)
#   4. Run data_pipeline.py → ml_forecasting.py for that ticker
#
# To retrain a ticker after drift:
#   1. Update TRAIN_END[ticker] to the new cutoff date
#   2. Run: python src/ml_forecasting.py (for that ticker only)
#   3. MODEL_REGISTRY updates automatically after retraining

import os
from datetime import datetime


# ── TICKERS ───────────────────────────────────────────────────────────

# Pre-trained tickers — models exist on disk, instant prediction in app.
# Any valid yfinance symbol can be added here after running the pipeline.
PRETRAINED_TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT"]

# ETFs have no earnings dates — earnings proximity features are skipped.
# Add any ETF ticker here to prevent earnings fetching errors.
ETF_TICKERS = ["SPY", "QQQ", "IWM", "VTI", "VOO"]

# Market benchmark — used for relative strength feature and portfolio comparison.
BENCHMARK = "SPY"


# ── TRAINING WINDOW ───────────────────────────────────────────────────

# Global training start — same for all tickers.
# Jan 2023 chosen to capture post-COVID normalization period
# and three complete annual earnings cycles through Feb 2026.
TRAIN_START = "2023-01-01"

# Per-ticker training end dates.
# Updated independently when drift is detected for a specific ticker.
# Retraining one ticker does not affect others.
# "default" is used for any ticker not explicitly listed —
# covers on-demand tickers requested by users at runtime.
TRAIN_END = {
    "AAPL":    "2026-02-28",
    "GOOGL":   "2026-02-28",
    "AMZN":    "2026-02-28",
    "MSFT":    "2026-02-28",
    "default": "2026-02-28",
}

# First date the app makes live predictions on.
# Everything from this date onward is inference territory —
# the model has never seen this data during training.
INFERENCE_START = "2026-03-01"

# How many months of recent history to fetch at inference time.
# Must cover the longest rolling window in feature engineering:
#   sma_50:        50 trading days  (~2.5 months)
#   vix_ma20:      20 trading days  (~1 month)
#   trends_zscore: 52 weeks         (~12 months)
# 14 months covers all of these safely with margin.
INFERENCE_LOOKBACK_MONTHS = 14


# ── MODEL REGISTRY ────────────────────────────────────────────────────

# Single source of truth for model state across all tickers.
# Updated automatically by ml_forecasting.py after each training run.
# baseline_accuracy: accuracy on held-out test set at training time.
# trained_on: ISO date string of when the model was last trained.
# train_end: the cutoff date used for this model version.
#
# Used by the app to:
#   - Display model metadata alongside predictions
#   - Compute drift = baseline_accuracy - live_accuracy
#   - Trigger retraining alert if drift > DRIFT_THRESHOLD
MODEL_REGISTRY = {
    "AAPL": {
        "baseline_accuracy": None,
        "trained_on":        None,
        "train_end":         "2026-02-28",
        "n_features":        None,
    },
    "GOOGL": {
        "baseline_accuracy": None,
        "trained_on":        None,
        "train_end":         "2026-02-28",
        "n_features":        None,
    },
    "AMZN": {
        "baseline_accuracy": None,
        "trained_on":        None,
        "train_end":         "2026-02-28",
        "n_features":        None,
    },
    "MSFT": {
        "baseline_accuracy": None,
        "trained_on":        None,
        "train_end":         "2026-02-28",
        "n_features":        None,
    },
}


def update_registry(ticker: str, accuracy: float, n_features: int) -> None:
    """
    Update MODEL_REGISTRY after a training run.

    Called automatically by ml_forecasting.py run_ml_pipeline().
    Records accuracy, feature count, training date, and cutoff date
    so the app always has current model metadata.

    Note: this updates the in-memory registry only.
    The registry is persisted to disk via save_registry() below.
    """
    MODEL_REGISTRY[ticker] = {
        "baseline_accuracy": round(accuracy, 4),
        "trained_on":        datetime.now().strftime("%Y-%m-%d"),
        "train_end":         get_train_end(ticker),
        "n_features":        n_features,
    }


def save_registry() -> None:
    """
    Persist MODEL_REGISTRY to data/model_registry.json.

    Called after update_registry() so model metadata survives
    across sessions and is readable by the Streamlit app.
    """
    import json
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "model_registry.json")
    with open(path, "w") as f:
        json.dump(MODEL_REGISTRY, f, indent=2)


def load_registry() -> dict:
    """
    Load MODEL_REGISTRY from disk if it exists.

    Called at app startup to restore model metadata from the last
    training run. Falls back to in-memory defaults if file not found.
    """
    import json
    path = os.path.join(DATA_DIR, "model_registry.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return MODEL_REGISTRY


# ── DRIFT DETECTION ───────────────────────────────────────────────────

# Accuracy drop that triggers a retraining alert in the app.
# Soft alert at 3% — monitor closely.
# Hard trigger at 5% — retrain recommended.
DRIFT_SOFT_ALERT  = 0.03
DRIFT_THRESHOLD   = 0.05

# Minimum number of live predictions required before drift is computed.
# Avoids false alerts from small sample noise in early inference period.
DRIFT_MIN_SAMPLES = 20


# ── ML SETTINGS ───────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_CV_SPLITS  = 5

# XGBoost hyperparameter grid.
# GridSearch tries every combination — 243 total.
# Extend carefully — each new value multiplies search time.
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


# ── GOOGLE TRENDS SETTINGS ────────────────────────────────────────────

# Trends resolution is standardized to weekly for both training and
# inference to avoid resolution mismatch.
# pytrends returns weekly data for multi-year queries automatically.
# At inference time we explicitly request weekly to match training.
TRENDS_RESOLUTION = "weekly"

# Custom search queries for pre-trained tickers.
# Company name gives better coverage than ticker symbol alone.
# Unknown tickers fall back to get_trends_query() dynamic default.
_TRENDS_QUERIES = {
    "AAPL":  "Apple stock",
    "GOOGL": "Google stock",
    "AMZN":  "Amazon stock",
    "MSFT":  "Microsoft stock",
    "SPY":   "S&P 500 ETF",
}


# ── GDELT SETTINGS ────────────────────────────────────────────────────

# GDELT DOC API — no authentication required.
# timelinetone mode returns average article sentiment tone per time bucket.
# Tone scale: negative values = negative coverage, positive = positive.
GDELT_API_URL     = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 10    # headlines fetched per API call for Gemini scoring
GDELT_MODE        = "timelinetone"  # returns sentiment over time

# Custom GDELT search queries for pre-trained tickers.
# More specific queries reduce irrelevant article noise.
_GDELT_QUERIES = {
    "AAPL":  "Apple Inc stock market",
    "GOOGL": "Google Alphabet stock market",
    "AMZN":  "Amazon stock market",
    "MSFT":  "Microsoft stock market",
    "SPY":   "S&P 500 index market",
}


# ── GEMINI SETTINGS ───────────────────────────────────────────────────

GEMINI_MODEL      = "gemini-2.5-flash"
GEMINI_BATCH_SIZE = 10     # headlines scored per API call
GEMINI_RATE_LIMIT = 1.0    # seconds between batches — respects free tier limits


# ── NEWS SETTINGS (live report only) ─────────────────────────────────

# News is fetched only at inference time for the research report.
# It is NOT used as a training feature due to historical data limitations.
# GDELT + Gemini handles training-time sentiment.
NEWS_LOOKBACK_DAYS   = 7
NEWSAPI_MAX_RESULTS  = 30
YFINANCE_MAX_RESULTS = 20


# ── PORTFOLIO OPTIMIZER SETTINGS ──────────────────────────────────────

# Approximate current 3-month T-bill rate used in Sharpe computation.
# Update this when interest rate environment changes significantly.
RISK_FREE_RATE = 0.04   # 4% annualized

# Mean-variance optimization constraints.
# Prevents degenerate solutions where optimizer puts everything in one asset.
MIN_WEIGHT = 0.05   # minimum 5% per asset
MAX_WEIGHT = 0.60   # maximum 60% per asset


# ── PATHS ─────────────────────────────────────────────────────────────

DATA_DIR    = "data"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
SRC_DIR     = "src"


def data_path(filename: str) -> str:
    """Full path to a file in the data directory."""
    return os.path.join(DATA_DIR, filename)


def model_path(ticker: str) -> str:
    """Full path to a saved model file for a ticker."""
    return os.path.join(MODELS_DIR, f"{ticker}_model.pkl")


def output_path(filename: str) -> str:
    """Full path to a file in the outputs directory."""
    return os.path.join(OUTPUTS_DIR, filename)


# ── DYNAMIC QUERY BUILDERS ────────────────────────────────────────────

def get_train_end(ticker: str) -> str:
    """
    Return the training cutoff date for a ticker.

    Pre-trained tickers have explicit dates in TRAIN_END.
    Unknown tickers (on-demand user requests) use the default cutoff.
    This ensures on-demand tickers train on the same window as
    pre-trained ones for fair comparison.
    """
    return TRAIN_END.get(ticker, TRAIN_END["default"])


def get_trends_query(ticker: str, company_name: str) -> str:
    """
    Return Google Trends search query for a ticker.

    Uses custom query for pre-trained tickers.
    Falls back to '{company_name} stock' for unknown tickers —
    works for any valid stock without hardcoding.
    """
    return _TRENDS_QUERIES.get(ticker, f"{company_name} stock")


def get_gdelt_query(ticker: str, company_name: str) -> str:
    """
    Return GDELT search query for a ticker.

    Uses custom query for pre-trained tickers.
    Falls back to '{company_name} stock market' for unknown tickers.
    """
    return _GDELT_QUERIES.get(ticker, f"{company_name} stock market")


def is_etf(ticker: str) -> bool:
    """
    Return True if ticker is an ETF.

    ETFs have no earnings dates — this flag prevents earnings
    proximity feature computation from running for ETF tickers,
    avoiding yfinance errors and empty feature columns.
    """
    return ticker.upper() in ETF_TICKERS


def is_pretrained(ticker: str) -> bool:
    """
    Return True if a model has been pre-trained for this ticker.

    Used by the Streamlit app to decide whether to show instant
    prediction or trigger the on-demand training pipeline.
    """
    return ticker.upper() in PRETRAINED_TICKERS