# src/data_pipeline.py
#
# Data fetching layer for the quant research assistant.
# Fetches and saves all data required for model training:
#   - OHLCV price data per ticker
#   - Macro context (VIX, TNX)
#   - Earnings announcement dates per ticker
#   - Google Trends attention signal per ticker
#   - GDELT sentiment scored by Gemini per ticker
#
# All data is saved to data/ as CSV files.
# ml_forecasting.py reads these files and merges them during
# feature engineering — data fetching and feature engineering
# are intentionally kept separate.
#
# Run this script once before training:
#   python src/data_pipeline.py
#
# Re-run when:
#   - Adding a new ticker to PRETRAINED_TICKERS in config.py
#   - Retraining after drift (update TRAIN_END in config.py first)

import os
import sys
import time
import requests
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from dotenv import load_dotenv

# Add project root to path so config.py is importable from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PRETRAINED_TICKERS,
    ETF_TICKERS,
    TRAIN_START,
    GEMINI_MODEL,
    GEMINI_BATCH_SIZE,
    GEMINI_RATE_LIMIT,
    GDELT_API_URL,
    GDELT_MAX_RECORDS,
    GDELT_MODE,
    TRENDS_RESOLUTION,
    DATA_DIR,
    data_path,
    get_train_end,
    get_trends_query,
    get_gdelt_query,
    is_etf,
)

warnings.filterwarnings("ignore")
load_dotenv()



# ── OHLCV ─────────────────────────────────────────────────────────────

def fetch_market_data(ticker: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker within the training window.

    Uses fixed TRAIN_START and per-ticker TRAIN_END from config
    rather than a relative period — ensures the training dataset
    is reproducible regardless of when the script is run.

    Args:
        ticker: valid yfinance symbol e.g. 'AAPL'

    Returns:
        DataFrame with OHLCV columns indexed by date.
    """
    train_end = get_train_end(ticker)
    print(f"  Fetching OHLCV for {ticker} ({TRAIN_START} → {train_end})...")

    raw = yf.download(
        ticker,
        start=TRAIN_START,
        end=train_end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}")

    # Flatten multi-level columns — yfinance quirk when downloading single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    print(f"  Retrieved {len(df)} rows for {ticker}")
    return df


# ── MACRO ─────────────────────────────────────────────────────────────

def fetch_macro_data() -> pd.DataFrame:
    """
    Fetch macro context features shared across all tickers.

    Fetches VIX and 10-year treasury yield (TNX) for the full
    training window. Macro data is ticker-independent — saved
    once and merged into every ticker's feature matrix.

    Derived features:
        vix_change : day-over-day VIX change — direction of fear
        tnx_change : day-over-day yield change — direction of rates
        vix_ma20   : 20-day smoothed VIX — volatility regime indicator
    """
    # Use the earliest possible end date across all tickers
    train_end = get_train_end("default")
    print(f"  Fetching macro data (VIX, TNX) ({TRAIN_START} → {train_end})...")

    vix = yf.download("^VIX", start=TRAIN_START, end=train_end,
                      interval="1d", auto_adjust=True, progress=False)
    tnx = yf.download("^TNX", start=TRAIN_START, end=train_end,
                      interval="1d", auto_adjust=True, progress=False)

    for df in [vix, tnx]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    macro = pd.DataFrame(index=vix.index)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)

    macro["vix"]        = vix["Close"]
    macro["tnx"]        = tnx["Close"]
    macro["vix_change"] = macro["vix"].pct_change()
    macro["tnx_change"] = macro["tnx"].pct_change()
    macro["vix_ma20"]   = macro["vix"].rolling(20).mean()

    macro.dropna(inplace=True)
    print(f"  Retrieved {len(macro)} rows of macro data")
    return macro


# ── EARNINGS ──────────────────────────────────────────────────────────

def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Fetch historical earnings announcement dates for a ticker.

    Skipped automatically for ETF tickers — they have no earnings.
    Returns empty DataFrame if data unavailable rather than raising.

    yfinance earnings_dates coverage varies — some tickers have
    complete history, others only recent quarters.
    """
    if is_etf(ticker):
        print(f"  {ticker} is an ETF — skipping earnings fetch")
        return pd.DataFrame(columns=["earnings_date"])

    print(f"  Fetching earnings dates for {ticker}...")
    try:
        t        = yf.Ticker(ticker)
        earnings = t.earnings_dates

        if earnings is None or earnings.empty:
            print(f"  No earnings data available for {ticker}")
            return pd.DataFrame(columns=["earnings_date"])

        dates = earnings.dropna(subset=["Reported EPS"]).index
        dates = pd.to_datetime(dates).tz_localize(None)

        # Filter to training window only
        train_end = pd.to_datetime(get_train_end(ticker))
        dates     = [d for d in dates
                     if pd.to_datetime(TRAIN_START) <= d <= train_end]

        df = pd.DataFrame({"earnings_date": sorted(dates)})
        print(f"  Retrieved {len(df)} earnings dates for {ticker}")
        return df

    except Exception as e:
        print(f"  Could not fetch earnings for {ticker}: {e}")
        return pd.DataFrame(columns=["earnings_date"])


# ── GOOGLE TRENDS ─────────────────────────────────────────────────────

def fetch_trends_data(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Fetch Google Trends attention signal for a ticker.

    Trends measures relative search interest (0-100) normalized
    to the peak within the requested period. It is an attention
    signal — not directional sentiment. High values indicate
    elevated retail investor attention which correlates with
    increased volatility.

    Resolution is standardized to weekly for both training and
    inference to avoid resolution mismatch. Weekly scores are
    forward-filled to daily in the feature engineering step.

    Derived features saved:
        trends_score   : raw weekly attention index (0-100)
        trends_change  : week-over-week change in attention
        trends_zscore  : standardized score vs 52-week baseline
                         high z-score = abnormally elevated attention
        trends_spike   : binary flag — z-score > 2.0
    """
    query     = get_trends_query(ticker, company_name)
    train_end = get_train_end(ticker)

    print(f"  Fetching Google Trends for '{query}' ({TRAIN_START} → {train_end})...")

    try:
        pytrends = TrendReq(hl="en-US", tz=360)

        # pytrends timeframe format: 'YYYY-MM-DD YYYY-MM-DD'
        timeframe = f"{TRAIN_START} {train_end}"
        pytrends.build_payload([query], timeframe=timeframe)
        raw = pytrends.interest_over_time()

        if raw.empty:
            print(f"  No trends data returned for {ticker}")
            return pd.DataFrame()

        # Drop the isPartial column — only present at end of series
        if "isPartial" in raw.columns:
            raw = raw.drop(columns=["isPartial"])

        df = pd.DataFrame(index=raw.index)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        df["trends_score"]  = raw[query].astype(float)
        df["trends_change"] = df["trends_score"].pct_change()

        # Z-score vs 52-week rolling baseline
        rolling_mean        = df["trends_score"].rolling(52, min_periods=4).mean()
        rolling_std         = df["trends_score"].rolling(52, min_periods=4).std()
        df["trends_zscore"] = (df["trends_score"] - rolling_mean) / rolling_std.replace(0, 1)

        # Binary spike indicator — abnormally high attention
        df["trends_spike"]  = (df["trends_zscore"] > 2.0).astype(int)

        df.dropna(inplace=True)
        print(f"  Retrieved {len(df)} weekly trends rows for {ticker}")
        return df

    except Exception as e:
        print(f"  Google Trends fetch failed for {ticker}: {e}")
        return pd.DataFrame()


# ── GDELT ─────────────────────────────────────────────────────────────

def fetch_gdelt_timelinetone(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Fetch pre-computed sentiment tone from GDELT DOC API timelinetone mode.

    timelinetone returns average article tone per time bucket across the
    full query period — no Gemini calls needed. GDELT computes tone using
    linguistic analysis across thousands of sources simultaneously.

    Tone scale: negative values = negative coverage, positive = positive.
    Typical range is roughly -10 to +10 though extremes can exceed this.

    Makes a single API call per ticker covering the full training window
    rather than monthly chunks — more reliable than artlist mode which
    frequently returns empty responses for historical date ranges.

    Derived features:
        gdelt_tone      : raw daily tone score
        gdelt_tone_ma7  : 7-day smoothed tone — reduces daily noise
        gdelt_tone_change: day-over-day tone change — sentiment momentum
        gdelt_positive   : 1 if tone > 0.5, else 0 — binary positive signal
        gdelt_negative   : 1 if tone < -0.5, else 0 — binary negative signal
    """
    query     = get_gdelt_query(ticker, company_name)
    train_end = get_train_end(ticker)

    # GDELT datetime format: YYYYMMDDHHMMSS
    start_dt = datetime.strptime(TRAIN_START, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt   = datetime.strptime(train_end, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    print(f"  Fetching GDELT timelinetone for '{query}'...")

    params = {
        "query":         query,
        "mode":          "timelinetone",
        "format":        "json",
        "startdatetime": start_dt,
        "enddatetime":   end_dt,
    }

    try:
        response = requests.get(GDELT_API_URL, params=params, timeout=30)

        if not response.text.strip():
            print(f"  GDELT returned empty response for {ticker}")
            return pd.DataFrame()

        data     = response.json()
        timeline = data.get("timeline", [])

        if not timeline:
            print(f"  No timeline data returned for {ticker}")
            return pd.DataFrame()

        # Each entry in timeline has a list of data points
        # Structure: [{"series": [{"date": "...", "value": tone}, ...]}]
        records = []
        for series in timeline:
            for point in series.get("data", []):
                raw_date = point.get("date", "")
                tone     = point.get("value", 0.0)

                try:
                    # GDELT date format varies — handle both YYYYMMDDTHHMMSSZ and YYYY-MM-DD
                    if "T" in raw_date:
                        date = datetime.strptime(raw_date[:8], "%Y%m%d").strftime("%Y-%m-%d")
                    else:
                        date = raw_date[:10]
                    records.append({"date": date, "gdelt_tone": float(tone)})
                except Exception:
                    continue

        if not records:
            print(f"  Could not parse GDELT timeline data for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df.index = df.index.tz_localize(None)

        # Remove duplicate dates — keep mean if multiple entries per day
        df = df.groupby(df.index).mean()

        # Derived features
        df["gdelt_tone_ma7"]    = df["gdelt_tone"].rolling(7, min_periods=1).mean()
        df["gdelt_tone_change"] = df["gdelt_tone"].diff()
        df["gdelt_positive"]    = (df["gdelt_tone"] > 0.5).astype(int)
        df["gdelt_negative"]    = (df["gdelt_tone"] < -0.5).astype(int)

        df.dropna(subset=["gdelt_tone"], inplace=True)

        print(f"  Retrieved {len(df)} tone data points for {ticker}")
        return df

    except requests.exceptions.Timeout:
        print(f"  GDELT request timed out for {ticker} — skipping")
        return pd.DataFrame()
    except Exception as e:
        print(f"  GDELT fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_gdelt_sentiment(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Orchestrate GDELT sentiment fetching for a ticker.

    Uses timelinetone mode — single API call per ticker, pre-computed
    tone scores, no Gemini quota consumed, reliable historical coverage.

    Caching: saves result to data/{ticker}_gdelt_daily.csv on success.
    Subsequent runs load from cache if file exists and is non-empty —
    avoids redundant API calls when rerunning the pipeline.

    Falls back gracefully if GDELT is unavailable — returns empty
    DataFrame and logs a warning. ml_forecasting.py handles missing
    GDELT files by skipping those feature columns rather than crashing.
    """
    cache_path = data_path(f"{ticker}_gdelt_daily.csv")

    # Load from cache if available
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not cached.empty:
            print(f"  Loaded GDELT sentiment from cache ({len(cached)} rows)")
            return cached

    # Fetch from API
    daily = fetch_gdelt_timelinetone(ticker, company_name)

    if daily.empty:
        print(f"  No GDELT data available for {ticker} — sentiment features will be skipped")
        return pd.DataFrame()

    # Save to cache
    daily.to_csv(cache_path)
    print(f"  Saved GDELT daily sentiment to {cache_path} ({len(daily)} rows)")

    return daily


# ── COMPANY NAME ──────────────────────────────────────────────────────

def get_company_name(ticker: str) -> str:
    """
    Fetch full company name for a ticker using yfinance.
    Used to build more effective GDELT and Trends search queries.
    Falls back to ticker symbol if unavailable.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


# ── SAVE ──────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to the data/ directory as CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = data_path(filename)
    df.to_csv(path)
    print(f"  Saved {filename} ({len(df)} rows)")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────

def run_data_pipeline(tickers: list[str] = None) -> None:
    """
    Run the full data fetching pipeline for all configured tickers.

    Fetching order:
        1. OHLCV per ticker
        2. Macro data (once — shared across tickers)
        3. Earnings dates per ticker (skipped for ETFs)
        4. Google Trends per ticker
        5. GDELT sentiment per ticker (slowest — uses caching)

    Args:
        tickers: list of ticker symbols to fetch.
                 Defaults to PRETRAINED_TICKERS from config.
    """
    if tickers is None:
        tickers = PRETRAINED_TICKERS

    print(f"\nStarting data pipeline for: {tickers}")
    print(f"Training window: {TRAIN_START} → {get_train_end('default')}\n")

    # Stage 1 — OHLCV per ticker
    print("=" * 50)
    print("Stage 1: OHLCV Data")
    print("=" * 50)
    for ticker in tickers:
        try:
            df = fetch_market_data(ticker)
            save_data(df, f"{ticker}_raw.csv")
        except Exception as e:
            print(f"  ERROR fetching OHLCV for {ticker}: {e}")

    # Stage 2 — Macro data (once)
    print("\n" + "=" * 50)
    print("Stage 2: Macro Data (VIX, TNX)")
    print("=" * 50)
    try:
        macro = fetch_macro_data()
        save_data(macro, "macro.csv")
    except Exception as e:
        print(f"  ERROR fetching macro data: {e}")

    # Stage 3 — Earnings per ticker
    print("\n" + "=" * 50)
    print("Stage 3: Earnings Dates")
    print("=" * 50)
    for ticker in tickers:
        try:
            earnings = fetch_earnings_dates(ticker)
            if not earnings.empty:
                save_data(earnings, f"{ticker}_earnings.csv")
        except Exception as e:
            print(f"  ERROR fetching earnings for {ticker}: {e}")

    # Stage 4 — Google Trends per ticker
    print("\n" + "=" * 50)
    print("Stage 4: Google Trends")
    print("=" * 50)
    for ticker in tickers:
        try:
            company_name = get_company_name(ticker)
            trends       = fetch_trends_data(ticker, company_name)
            if not trends.empty:
                save_data(trends, f"{ticker}_trends.csv")
        except Exception as e:
            print(f"  ERROR fetching trends for {ticker}: {e}")

    # Stage 5 — GDELT sentiment per ticker (slowest)
    print("\n" + "=" * 50)
    print("Stage 5: GDELT Sentiment (Gemini-scored)")
    print("=" * 50)
    print("Note: First run fetches and scores 3 years of headlines.")
    print("      Subsequent runs load from cache — much faster.\n")
    for ticker in tickers:
        try:
            company_name = get_company_name(ticker)
            print(f"Processing GDELT for {ticker} ({company_name})...")
            fetch_gdelt_sentiment(ticker, company_name)
        except Exception as e:
            print(f"  ERROR processing GDELT for {ticker}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("Data pipeline complete. Files saved:")
    print("=" * 50)
    for f in sorted(os.listdir(DATA_DIR)):
        path = data_path(f)
        size = os.path.getsize(path) / 1024
        print(f"  {f:<40} {size:>8.1f} KB")


if __name__ == "__main__":
    run_data_pipeline()