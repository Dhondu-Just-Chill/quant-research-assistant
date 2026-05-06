# config.py
# Central configuration for the quant research assistant.
# All other modules import from here — adding a ticker or changing
# a date requires editing only this file.

import os

# ── TICKERS ───────────────────────────────────────────────────────────

# Tickers to train models for.
# Any valid yfinance symbol can be added here.
# SPY is included for benchmark comparison but excluded from
# earnings proximity features since it is an ETF.
TICKERS     = ["AAPL", "GOOGL", "AMZN", "MSFT"]
BENCHMARK   = "SPY"
ETF_TICKERS = ["SPY"]   # tickers that have no earnings dates


# ── TRAINING WINDOW ───────────────────────────────────────────────────

# Fixed cutoff dates — do not change unless retraining.
# Training ends at Feb 2026 to capture 3 complete earnings cycles
# (Jan 2023 → Feb 2026) and align with the post-Q4 earnings calendar.
# Inference runs on live data from INFERENCE_START onward.
TRAIN_START     = "2023-01-01"
TRAIN_END       = "2026-02-28"
INFERENCE_START = "2026-03-01"   # first date the app predicts on

# How many months of recent history to fetch at inference time.
# Must be long enough to compute all rolling window features:
#   sma_50:       50 days
#   vix_ma20:     20 days
#   trends_zscore: 52 weeks
# 6 months covers all of these safely.
INFERENCE_LOOKBACK_MONTHS = 6


# ── DRIFT DETECTION ───────────────────────────────────────────────────

# Accuracy drop threshold that triggers a retraining alert.
# If live accuracy falls more than this below training baseline,
# the model should be retrained with an extended training window.
DRIFT_THRESHOLD = 0.05   # 5 percentage points


# ── ML SETTINGS ───────────────────────────────────────────────────────

TEST_SIZE    = 0.2    # fraction of data held out for evaluation
RANDOM_STATE = 42     # seed for reproducibility across all models
N_CV_SPLITS  = 5      # number of TimeSeriesSplit folds in GridSearch

# XGBoost hyperparameter search grid.
# GridSearch tries every combination — adding values here increases
# search time quadratically so keep the grid focused.
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


# ── GOOGLE TRENDS SETTINGS ────────────────────────────────────────────

# Trends resolution is weekly for multi-year queries.
# We standardize to weekly for both training and inference
# to avoid resolution mismatch between training and live data.
TRENDS_RESOLUTION = "weekly"

# Search queries per ticker — company name gives better coverage
# than ticker symbol alone for Google Trends.
TRENDS_QUERIES = {
    "AAPL":  "Apple stock",
    "GOOGL": "Google stock",
    "AMZN":  "Amazon stock",
    "MSFT":  "Microsoft stock",
    "SPY":   "S&P 500 stock",
}


# ── GDELT SETTINGS ────────────────────────────────────────────────────

# GDELT DOC API base URL — no authentication required.
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Number of headlines to fetch per GDELT request.
# Higher = richer sentiment signal but more Gemini API calls.
GDELT_MAX_RECORDS = 10

# GDELT search queries per ticker — same logic as trends queries.
GDELT_QUERIES = {
    "AAPL":  "Apple Inc stock market",
    "GOOGL": "Google Alphabet stock market",
    "AMZN":  "Amazon stock market",
    "MSFT":  "Microsoft stock market",
    "SPY":   "S&P 500 index market",
}


# ── GEMINI SETTINGS ───────────────────────────────────────────────────

GEMINI_MODEL      = "gemini-2.5-flash"
GEMINI_BATCH_SIZE = 10     # headlines per Gemini API call
GEMINI_RATE_LIMIT = 1.0    # seconds to sleep between batches (free tier)


# ── PATHS ─────────────────────────────────────────────────────────────

# All paths relative to project root.
# Directories are created automatically by each module if missing.
DATA_DIR    = "data"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"

def data_path(filename: str) -> str:
    """Return full path to a file in the data directory."""
    return os.path.join(DATA_DIR, filename)

def model_path(ticker: str) -> str:
    """Return full path to a saved model file."""
    return os.path.join(MODELS_DIR, f"{ticker}_model.pkl")

def output_path(filename: str) -> str:
    """Return full path to a file in the outputs directory."""
    return os.path.join(OUTPUTS_DIR, filename)


# ── PORTFOLIO OPTIMIZER SETTINGS ──────────────────────────────────────

# Risk-free rate used in Sharpe ratio calculation.
# Approximate current 3-month T-bill rate.
RISK_FREE_RATE = 0.04   # 4% annualized

# Constraints for mean-variance optimization.
MIN_WEIGHT = 0.05   # no single asset below 5% allocation
MAX_WEIGHT = 0.60   # no single asset above 60% allocation


# ── NEWS SETTINGS ─────────────────────────────────────────────────────

# Number of days of recent news to fetch for live report generation.
NEWS_LOOKBACK_DAYS = 7

# Maximum headlines to fetch per source for report generation.
NEWSAPI_MAX_RESULTS   = 30
YFINANCE_MAX_RESULTS  = 20