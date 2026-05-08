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
import json
import time
import requests
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from google import genai
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

# Initialize Gemini client for GDELT headline scoring
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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

def fetch_gdelt_headlines(ticker: str, company_name: str,
                          start: str, end: str) -> list[dict]:
    """
    Fetch news headlines from the GDELT DOC API for a date range.

    Uses the artlist mode to retrieve individual article headlines
    rather than pre-aggregated tone scores — this lets us pipe
    headlines through Gemini for financially-aware scoring rather
    than relying on GDELT's dictionary-based tone computation.

    Args:
        ticker       : stock symbol
        company_name : full company name for better query coverage
        start        : start date string 'YYYY-MM-DD'
        end          : end date string 'YYYY-MM-DD'

    Returns list of {headline, date, source} dicts.
    """
    query = get_gdelt_query(ticker, company_name)

    # GDELT datetime format: YYYYMMDDHHMMSS
    start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt   = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    GDELT_MAX_RECORDS,
        "startdatetime": start_dt,
        "enddatetime":   end_dt,
        "format":        "json",
        "sort":          "DateDesc",
    }

    try:
        response = requests.get(GDELT_API_URL, params=params, timeout=15)
        data     = response.json()
        articles = data.get("articles", [])

        results = []
        for article in articles:
            # GDELT date format: YYYYMMDDTHHMMSSZ
            raw_date = article.get("seendate", "")
            try:
                date = datetime.strptime(
                    raw_date[:8], "%Y%m%d"
                ).strftime("%Y-%m-%d")
            except Exception:
                date = start

            results.append({
                "headline": article.get("title", ""),
                "date":     date,
                "source":   article.get("domain", "gdelt"),
            })

        return results

    except Exception as e:
        print(f"    GDELT fetch error ({start} → {end}): {e}")
        return []


def score_headlines_gemini(headlines: list[dict],
                           ticker: str) -> list[dict]:
    """
    Score a batch of headlines using Gemini for financial sentiment.

    Gemini provides financially-aware scoring that understands context:
        "beats estimates" → positive even without positive words
        "misses revenue but raises guidance" → net positive
        "faces antitrust probe" → negative regardless of neutral wording

    Each headline scored on:
        sentiment : positive / negative / neutral
        score     : float -1.0 to +1.0
        relevance : high / medium / low

    Low-relevance headlines excluded from daily aggregate in
    compute_daily_sentiment() to reduce noise.
    """
    if not headlines:
        return []

    scored = []
    for i in range(0, len(headlines), GEMINI_BATCH_SIZE):
        batch = headlines[i:i + GEMINI_BATCH_SIZE]

        headlines_text = "\n".join([
            f"{j+1}. {h['headline']}"
            for j, h in enumerate(batch)
        ])

        prompt = f"""You are a financial sentiment analyst.
Score each headline's sentiment toward {ticker} stock.

Headlines:
{headlines_text}

Return ONLY a JSON array. No markdown, no backticks, no preamble.
Each element must have:
  "index"     : headline number (1-based integer)
  "sentiment" : "positive" | "negative" | "neutral"
  "score"     : float from -1.0 (very negative) to 1.0 (very positive)
  "relevance" : "high" | "medium" | "low"

Example: [{{"index": 1, "sentiment": "positive", "score": 0.7, "relevance": "high"}}]"""

        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            text = response.text.strip()

            # Strip markdown fences if Gemini adds them despite instructions
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0]

            batch_scores = json.loads(text.strip())

            for item in batch_scores:
                idx = item["index"] - 1
                if 0 <= idx < len(batch):
                    scored.append({
                        **batch[idx],
                        "sentiment": item.get("sentiment", "neutral"),
                        "score":     float(item.get("score", 0.0)),
                        "relevance": item.get("relevance", "medium"),
                    })

            time.sleep(GEMINI_RATE_LIMIT)

        except Exception as e:
            print(f"    Gemini scoring error: {e}")
            for item in batch:
                scored.append({
                    **item,
                    "sentiment": "neutral",
                    "score":     0.0,
                    "relevance": "low",
                })

    return scored


def compute_daily_sentiment(scored: list[dict]) -> pd.DataFrame:
    """
    Aggregate Gemini-scored headlines into daily sentiment scores.

    Only high and medium relevance headlines contribute to the
    daily aggregate — filters noise from tangentially related articles.

    Derived features:
        gdelt_score    : mean sentiment score for the day (-1 to +1)
        gdelt_ma7      : 7-day smoothed score — reduces daily noise
        gdelt_change   : day-over-day score change — sentiment momentum
        gdelt_positive : fraction of positive headlines that day
        gdelt_negative : fraction of negative headlines that day
    """
    if not scored:
        return pd.DataFrame()

    # Filter to relevant headlines
    relevant = [
        h for h in scored
        if h.get("relevance") in ["high", "medium"]
    ] or scored  # fallback to all if none flagged relevant

    records = []
    df_raw  = pd.DataFrame(relevant)

    for date, group in df_raw.groupby("date"):
        scores = group["score"].tolist()
        n      = len(scores)
        records.append({
            "date":           date,
            "gdelt_score":    float(np.mean(scores)),
            "gdelt_positive": sum(1 for s in scores if s > 0.1) / n,
            "gdelt_negative": sum(1 for s in scores if s < -0.1) / n,
        })

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"])
    result         = result.set_index("date").sort_index()
    result.index   = result.index.tz_localize(None)

    # Smoothed score and momentum
    result["gdelt_ma7"]    = result["gdelt_score"].rolling(7, min_periods=1).mean()
    result["gdelt_change"] = result["gdelt_score"].diff()

    return result


def fetch_gdelt_sentiment(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Orchestrate full GDELT sentiment pipeline for a ticker.

    Fetches headlines in monthly chunks across the training window
    to stay within GDELT API limits and avoid timeouts.

    Implements caching — if scored headlines already exist on disk
    for a given month, skips that month entirely. This means:
        - First run: fetches and scores all 3 years (slow, ~45 min)
        - Subsequent runs: loads from cache instantly
        - Partial runs: resumes from last completed month

    Saves two files:
        data/{ticker}_gdelt_raw.csv    : all scored headlines
        data/{ticker}_gdelt_daily.csv  : daily aggregated scores
    """
    train_end  = get_train_end(ticker)
    cache_path = data_path(f"{ticker}_gdelt_raw.csv")

    # Load existing cache if present
    existing_scored = []
    cached_dates    = set()

    if os.path.exists(cache_path):
        cached_df    = pd.read_csv(cache_path, parse_dates=["date"])
        existing_scored = cached_df.to_dict("records")
        cached_dates = set(
            pd.to_datetime(cached_df["date"]).dt.to_period("M").astype(str)
        )
        print(f"  Loaded {len(existing_scored)} cached headlines for {ticker}")

    # Generate monthly date ranges across training window
    start_dt = datetime.strptime(TRAIN_START, "%Y-%m-%d")
    end_dt   = datetime.strptime(train_end, "%Y-%m-%d")

    months = []
    current = start_dt
    while current <= end_dt:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        months.append((
            current.strftime("%Y-%m-%d"),
            min(next_month - timedelta(days=1), end_dt).strftime("%Y-%m-%d")
        ))
        current = next_month

    # Fetch and score month by month — skip cached months
    new_scored = []
    for month_start, month_end in months:
        period_key = pd.to_datetime(month_start).to_period("M").strftime("%Y-%m")

        if period_key in cached_dates:
            continue  # already scored — skip

        print(f"    Fetching GDELT {month_start} → {month_end}...")
        headlines = fetch_gdelt_headlines(ticker, company_name, month_start, month_end)

        if headlines:
            scored = score_headlines_gemini(headlines, ticker)
            new_scored.extend(scored)

        # Brief pause between months to respect GDELT rate limits
        time.sleep(0.5)

    # Combine cached + new scored headlines
    all_scored = existing_scored + new_scored

    if not all_scored:
        print(f"  No GDELT headlines found for {ticker}")
        return pd.DataFrame()

    # Save raw scored headlines — overwrites with full combined dataset
    raw_df = pd.DataFrame(all_scored)
    raw_df.to_csv(cache_path, index=False)
    print(f"  Saved {len(raw_df)} total scored headlines to {cache_path}")

    # Compute and save daily aggregates
    daily    = compute_daily_sentiment(all_scored)
    daily_path = data_path(f"{ticker}_gdelt_daily.csv")
    daily.to_csv(daily_path)
    print(f"  Saved daily sentiment to {daily_path}")

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