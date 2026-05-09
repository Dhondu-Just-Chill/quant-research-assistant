# config.py
#
# Central configuration for the quant research assistant.
# Single source of truth for all settings across the project.
#
# To add a new pre-trained ticker:
#   1. Add to PRETRAINED_TICKERS
#   2. Add to TRAIN_END if cutoff differs from default
#   3. Optionally add custom queries to _GDELT_QUERY_MAP
#   4. Run data_pipeline.py → ml_forecasting.py for that ticker
#
# To retrain a ticker after drift:
#   1. Update TRAIN_END[ticker] to new cutoff date
#   2. Run: python src/ml_forecasting.py (for that ticker only)
#   3. MODEL_REGISTRY updates automatically

import os
import re
from datetime import datetime


# ── TICKERS ───────────────────────────────────────────────────────────

PRETRAINED_TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT"]

# ETFs have no earnings dates — earnings proximity features are skipped.
ETF_TICKERS = ["SPY", "QQQ", "IWM", "VTI", "VOO"]

BENCHMARK = "SPY"


# ── TRAINING WINDOW ───────────────────────────────────────────────────

TRAIN_START = "2023-01-01"

# Per-ticker training end dates — updated independently on drift.
TRAIN_END = {
    "AAPL":    "2026-02-28",
    "GOOGL":   "2026-02-28",
    "AMZN":    "2026-02-28",
    "MSFT":    "2026-02-28",
    "default": "2026-02-28",
}

# First date the app predicts on — model has never seen this data.
INFERENCE_START = "2026-03-01"

# Months of history fetched at inference time to compute rolling features.
# 14 months covers the longest window (trends_zscore = 52 weeks).
INFERENCE_LOOKBACK_MONTHS = 14


# ── MODEL REGISTRY ────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "AAPL":  {"baseline_accuracy": None, "trained_on": None,
              "train_end": "2026-02-28", "n_features": None},
    "GOOGL": {"baseline_accuracy": None, "trained_on": None,
              "train_end": "2026-02-28", "n_features": None},
    "AMZN":  {"baseline_accuracy": None, "trained_on": None,
              "train_end": "2026-02-28", "n_features": None},
    "MSFT":  {"baseline_accuracy": None, "trained_on": None,
              "train_end": "2026-02-28", "n_features": None},
}


def update_registry(ticker: str, accuracy: float, n_features: int) -> None:
    """Update MODEL_REGISTRY after a training run."""
    MODEL_REGISTRY[ticker] = {
        "baseline_accuracy": round(accuracy, 4),
        "trained_on":        datetime.now().strftime("%Y-%m-%d"),
        "train_end":         get_train_end(ticker),
        "n_features":        n_features,
    }


def save_registry() -> None:
    """Persist MODEL_REGISTRY to data/model_registry.json."""
    import json
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "model_registry.json")
    with open(path, "w") as f:
        json.dump(MODEL_REGISTRY, f, indent=2)


def load_registry() -> dict:
    """Load MODEL_REGISTRY from disk — falls back to in-memory defaults."""
    import json
    path = os.path.join(DATA_DIR, "model_registry.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return MODEL_REGISTRY


# ── DRIFT DETECTION ───────────────────────────────────────────────────

DRIFT_SOFT_ALERT  = 0.03   # 3% drop — monitor closely
DRIFT_THRESHOLD   = 0.05   # 5% drop — retrain recommended
DRIFT_MIN_SAMPLES = 20     # minimum live predictions before drift computed


# ── ML SETTINGS ───────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_CV_SPLITS  = 5

PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


# ── FINBERT SETTINGS ──────────────────────────────────────────────────

# FinBERT is a BERT model fine-tuned on financial text.
# Runs locally — no API calls, no quota, deterministic output.
# Downloads ~440MB on first use, cached by HuggingFace permanently.
FINBERT_MODEL      = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32     # headlines per inference batch
FINBERT_MAX_LENGTH = 512    # max tokens — FinBERT's context limit

# Sentiment score mapping from FinBERT labels to numeric scores.
# Positive = +1.0, Negative = -1.0, Neutral = 0.0
# Multiplied by confidence score to produce a continuous score per headline.
FINBERT_SCORE_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral":  0.0,
}


# ── GDELT SETTINGS ────────────────────────────────────────────────────

GDELT_API_URL      = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS  = 15     # headlines per artlist request — enough for FinBERT
GDELT_TIMEOUT      = 30     # seconds before request is abandoned

# Sentiment layer weights for composite score.
# Company news is most directly relevant to stock price.
# Sector news captures industry-level events like supply chain shocks.
# XGBoost also receives both scores independently — can learn its own weights.
GDELT_COMPANY_WEIGHT = 0.6
GDELT_SECTOR_WEIGHT  = 0.4

# Pre-trained ticker query map — nested by layer.
# company: directly about this company
# sector:  industry/supply-chain events that affect this company
#
# Query design principle: use terms financial journalists actually write
# in headlines — product names, segments, events — not search-engine terms
# like "stock" or "shares" which GDELT's article index rarely matches.
_GDELT_QUERY_MAP = {
    "AAPL": {
        "company": [
            "Apple earnings",
            "Apple revenue",
            "Apple iPhone",
        ],
        "sector": [
            "semiconductor supply chain",
            "smartphone market",
            "consumer electronics demand",
        ],
    },
    "GOOGL": {
        "company": [
            "Alphabet earnings",
            "Google revenue",
            "Google advertising",
        ],
        "sector": [
            "digital advertising market",
            "cloud computing competition",
            "AI regulation antitrust",
        ],
    },
    "AMZN": {
        "company": [
            "Amazon earnings",
            "Amazon revenue",
            "Amazon AWS",
        ],
        "sector": [
            "e-commerce retail outlook",
            "cloud computing demand",
            "logistics supply chain",
        ],
    },
    "MSFT": {
        "company": [
            "Microsoft earnings",
            "Microsoft revenue",
            "Microsoft Azure",
        ],
        "sector": [
            "enterprise software demand",
            "cloud computing competition",
            "AI technology investment",
        ],
    },
}


# ── GOOGLE TRENDS SETTINGS ────────────────────────────────────────────

TRENDS_RESOLUTION = "weekly"

_TRENDS_QUERIES = {
    "AAPL":  "Apple stock",
    "GOOGL": "Google stock",
    "AMZN":  "Amazon stock",
    "MSFT":  "Microsoft stock",
    "SPY":   "S&P 500 ETF",
}


# ── NEWS SETTINGS (live report only) ─────────────────────────────────

NEWS_LOOKBACK_DAYS   = 7
NEWSAPI_MAX_RESULTS  = 30
YFINANCE_MAX_RESULTS = 20


# ── PORTFOLIO OPTIMIZER SETTINGS ──────────────────────────────────────

RISK_FREE_RATE = 0.04
MIN_WEIGHT     = 0.05
MAX_WEIGHT     = 0.60


# ── PATHS ─────────────────────────────────────────────────────────────

DATA_DIR    = "data"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
SRC_DIR     = "src"


def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

def model_path(ticker: str) -> str:
    return os.path.join(MODELS_DIR, f"{ticker}_model.pkl")

def output_path(filename: str) -> str:
    return os.path.join(OUTPUTS_DIR, filename)


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────

def get_train_end(ticker: str) -> str:
    """Return training cutoff date for a ticker."""
    return TRAIN_END.get(ticker, TRAIN_END["default"])


def get_trends_query(ticker: str, company_name: str) -> str:
    """Return Google Trends query — custom or dynamic default."""
    return _TRENDS_QUERIES.get(ticker, f"{company_name} stock")


def _clean_company_name(name: str) -> str:
    """
    Strip legal suffixes from company names for cleaner GDELT queries.
    'Tesla, Inc.' → 'Tesla'
    'Alphabet Inc.' → 'Alphabet'
    """
    suffixes = [
        r",?\s+Inc\.?", r",?\s+Corp\.?", r",?\s+Ltd\.?",
        r",?\s+LLC\.?", r",?\s+PLC\.?", r",?\s+Co\.?",
        r",?\s+Group\.?", r",?\s+Holdings?\.?",
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    return name.strip()


def generate_gdelt_queries(ticker: str, company_name: str) -> dict:
    """
    Auto-generate GDELT query layers for any ticker using yfinance metadata.

    Used as fallback when ticker is not in _GDELT_QUERY_MAP — enables
    ticker-agnostic operation without hardcoding every possible stock.

    Company layer: uses cleaned company name + key business terms.
    Sector layer:  uses yfinance sector/industry to capture Ring 2 news
                   (industry events like supply chain shocks, regulation).

    Args:
        ticker       : stock symbol e.g. 'TSLA'
        company_name : full name from yfinance e.g. 'Tesla, Inc.'

    Returns dict with 'company' and 'sector' query lists.
    """
    import yfinance as yf

    try:
        info       = yf.Ticker(ticker).info
        short_name = _clean_company_name(
            info.get("shortName") or company_name
        )
        industry   = info.get("industry", "")
        sector     = info.get("sector", "")

        company_queries = [
            f"{short_name} earnings",
            f"{short_name} revenue",
            f"{short_name} {industry}".strip() if industry else f"{short_name} results",
        ]

        # Sector queries — capture industry-level events
        sector_queries = []
        if industry:
            sector_queries.append(f"{industry} industry outlook")
        if sector:
            sector_queries.append(f"{sector} sector news")
        if not sector_queries:
            sector_queries = [f"{short_name} industry news"]

        return {
            "company": company_queries,
            "sector":  sector_queries,
        }

    except Exception:
        # Minimal fallback — always returns something usable
        short = _clean_company_name(company_name)
        return {
            "company": [f"{short} earnings", f"{short} revenue"],
            "sector":  [f"{short} industry"],
        }


def get_gdelt_queries(ticker: str, company_name: str) -> dict:
    """
    Return GDELT query layers for a ticker.

    Pre-trained tickers use handcrafted queries in _GDELT_QUERY_MAP —
    these are tuned to return reliable GDELT coverage.

    Unknown tickers (on-demand user requests) use generate_gdelt_queries()
    which auto-builds queries from yfinance metadata — fully automatic,
    no code changes needed for any valid ticker.

    Returns dict: {"company": [...], "sector": [...]}
    """
    if ticker in _GDELT_QUERY_MAP:
        return _GDELT_QUERY_MAP[ticker]
    return generate_gdelt_queries(ticker, company_name)


def is_etf(ticker: str) -> bool:
    """Return True if ticker is an ETF — skips earnings fetching."""
    return ticker.upper() in ETF_TICKERS


def is_pretrained(ticker: str) -> bool:
    """Return True if a model has been pre-trained for this ticker."""
    return ticker.upper() in PRETRAINED_TICKERS