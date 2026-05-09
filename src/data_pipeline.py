# src/data_pipeline.py
#
# Data fetching layer for the quant research assistant.
#
# Fetches and saves all data required for model training:
#   Stage 1 — OHLCV price data per ticker
#   Stage 2 — Macro context (VIX, TNX) — shared across tickers
#   Stage 3 — Earnings announcement dates per ticker
#   Stage 4 — Google Trends attention signal per ticker
#   Stage 5 — GDELT sentiment scored by FinBERT per ticker
#
# GDELT pipeline architecture:
#   - artlist mode fetches individual headlines with dates
#   - FinBERT scores each headline locally (no API quota)
#   - Falls back to timelinetone if artlist returns empty for a month
#   - Two sentiment layers: company-specific + sector-level
#   - Results cached to disk — subsequent runs skip already-scored months
#
# Run once before training:
#   python src/data_pipeline.py

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PRETRAINED_TICKERS,
    TRAIN_START,
    GDELT_API_URL,
    GDELT_MAX_RECORDS,
    GDELT_TIMEOUT,
    GDELT_COMPANY_WEIGHT,
    GDELT_SECTOR_WEIGHT,
    FINBERT_MODEL,
    FINBERT_BATCH_SIZE,
    FINBERT_MAX_LENGTH,
    FINBERT_SCORE_MAP,
    TRENDS_RESOLUTION,
    DATA_DIR,
    data_path,
    get_train_end,
    get_trends_query,
    get_gdelt_queries,
    is_etf,
)

warnings.filterwarnings("ignore")

# ── FINBERT INITIALIZATION ────────────────────────────────────────────

# Loaded once at module level — expensive to reload per call.
# HuggingFace downloads model on first run (~440MB), cached permanently.
_finbert_pipeline = None

def get_finbert():
    """
    Lazy-load FinBERT pipeline — only initializes on first call.

    Lazy loading means importing data_pipeline.py doesn't trigger
    a 440MB download — FinBERT only loads when sentiment scoring
    is actually needed.
    """
    global _finbert_pipeline
    if _finbert_pipeline is None:
        print("  Loading FinBERT model (first run downloads ~440MB)...")
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            task="text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            max_length=FINBERT_MAX_LENGTH,
            truncation=True,
            device=-1,    # CPU — set to 0 for GPU if available
        )
        print("  FinBERT loaded successfully")
    return _finbert_pipeline


# ── OHLCV ─────────────────────────────────────────────────────────────

def fetch_market_data(ticker: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data within the fixed training window.

    Uses TRAIN_START and per-ticker TRAIN_END from config — ensures
    the dataset is reproducible regardless of when the script is run.
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

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    print(f"  Retrieved {len(df)} rows")
    return df


# ── MACRO ─────────────────────────────────────────────────────────────

def fetch_macro_data() -> pd.DataFrame:
    """
    Fetch VIX and 10-year treasury yield for the full training window.

    Macro data is ticker-independent — fetched once, merged into every
    ticker's feature matrix during ml_forecasting.py.
    """
    train_end = get_train_end("default")
    print(f"  Fetching VIX, TNX ({TRAIN_START} → {train_end})...")

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
    print(f"  Retrieved {len(macro)} rows")
    return macro


# ── EARNINGS ──────────────────────────────────────────────────────────

def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Fetch historical earnings announcement dates for a ticker.
    Skipped for ETFs — they have no earnings.
    """
    if is_etf(ticker):
        print(f"  {ticker} is ETF — skipping earnings")
        return pd.DataFrame(columns=["earnings_date"])

    print(f"  Fetching earnings dates for {ticker}...")
    try:
        t        = yf.Ticker(ticker)
        earnings = t.earnings_dates

        if earnings is None or earnings.empty:
            return pd.DataFrame(columns=["earnings_date"])

        dates     = earnings.dropna(subset=["Reported EPS"]).index
        dates     = pd.to_datetime(dates).tz_localize(None)
        train_end = pd.to_datetime(get_train_end(ticker))
        dates     = [d for d in dates
                     if pd.to_datetime(TRAIN_START) <= d <= train_end]

        df = pd.DataFrame({"earnings_date": sorted(dates)})
        print(f"  Retrieved {len(df)} earnings dates")
        return df

    except Exception as e:
        print(f"  Earnings fetch failed: {e}")
        return pd.DataFrame(columns=["earnings_date"])


# ── GOOGLE TRENDS ─────────────────────────────────────────────────────

def fetch_trends_data(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Fetch Google Trends attention signal for a ticker.

    Measures relative search interest (0-100) — an attention signal,
    not directional sentiment. Standardized to weekly resolution for
    both training and inference to avoid resolution mismatch.

    Features:
        trends_score   : raw weekly index (0-100)
        trends_change  : week-over-week change
        trends_zscore  : deviation from 52-week baseline
        trends_spike   : binary flag — zscore > 2.0
    """
    query     = get_trends_query(ticker, company_name)
    train_end = get_train_end(ticker)
    print(f"  Fetching Trends for '{query}'...")

    try:
        pytrends  = TrendReq(hl="en-US", tz=360)
        timeframe = f"{TRAIN_START} {train_end}"
        pytrends.build_payload([query], timeframe=timeframe)
        raw = pytrends.interest_over_time()

        if raw.empty:
            return pd.DataFrame()

        if "isPartial" in raw.columns:
            raw = raw.drop(columns=["isPartial"])

        df = pd.DataFrame(index=raw.index)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["trends_score"]  = raw[query].astype(float)
        df["trends_change"] = df["trends_score"].pct_change()

        rolling_mean        = df["trends_score"].rolling(52, min_periods=4).mean()
        rolling_std         = df["trends_score"].rolling(52, min_periods=4).std()
        df["trends_zscore"] = (df["trends_score"] - rolling_mean) / rolling_std.replace(0, 1)
        df["trends_spike"]  = (df["trends_zscore"] > 2.0).astype(int)

        df.dropna(inplace=True)
        print(f"  Retrieved {len(df)} weekly rows")
        return df

    except Exception as e:
        print(f"  Trends fetch failed: {e}")
        return pd.DataFrame()


# ── GDELT HEADLINE FETCHING ───────────────────────────────────────────

def fetch_gdelt_artlist(query: str, start: str, end: str) -> list[dict]:
    """
    Fetch individual article headlines from GDELT artlist mode.

    Returns list of {headline, date} dicts for a single query
    over a specific date range. Called per-query per-month by
    fetch_gdelt_layer() which aggregates across queries.

    Falls back gracefully on empty response or timeout — returns
    empty list so the caller can fall back to timelinetone.
    """
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
        response = requests.get(GDELT_API_URL, params=params,
                                timeout=GDELT_TIMEOUT)
        if not response.text.strip():
            return []

        data     = response.json()
        articles = data.get("articles", [])

        results = []
        for article in articles:
            raw_date = article.get("seendate", "")
            try:
                date = datetime.strptime(raw_date[:8], "%Y%m%d").strftime("%Y-%m-%d")
            except Exception:
                date = start
            headline = article.get("title", "").strip()
            if headline:
                results.append({"headline": headline, "date": date})

        return results

    except Exception:
        return []


def fetch_gdelt_timelinetone(query: str, start: str, end: str) -> pd.Series:
    """
    Fetch pre-aggregated tone from GDELT timelinetone mode.

    Used as fallback when artlist returns empty for a month.
    Returns a Series indexed by date with pre-computed tone values.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt   = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    params = {
        "query":         query,
        "mode":          "timelinetone",
        "format":        "json",
        "startdatetime": start_dt,
        "enddatetime":   end_dt,
    }

    try:
        response = requests.get(GDELT_API_URL, params=params,
                                timeout=GDELT_TIMEOUT)
        if not response.text.strip():
            return pd.Series(dtype=float)

        data     = response.json()
        timeline = data.get("timeline", [])
        records  = []

        for series in timeline:
            for point in series.get("data", []):
                raw_date = point.get("date", "")
                tone     = point.get("value", 0.0)
                try:
                    date = datetime.strptime(raw_date[:8], "%Y%m%d").strftime("%Y-%m-%d") \
                           if "T" in raw_date else raw_date[:10]
                    records.append({"date": date, "tone": float(tone)})
                except Exception:
                    continue

        if not records:
            return pd.Series(dtype=float)

        df   = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df   = df.set_index("date").sort_index()
        df.index = df.index.tz_localize(None)
        return df.groupby(df.index)["tone"].mean()

    except Exception:
        return pd.Series(dtype=float)


# ── FINBERT SCORING ───────────────────────────────────────────────────

def score_with_finbert(headlines: list[dict]) -> list[dict]:
    """
    Score a list of headlines using FinBERT locally.

    FinBERT outputs three labels: positive, negative, neutral.
    We convert to a continuous score by multiplying the label's
    direction (+1/-1/0) by the model's confidence score.

    This gives scores in range (-1.0, +1.0):
        confident positive → score near +1.0
        confident negative → score near -1.0
        uncertain or neutral → score near 0.0

    Processes in batches of FINBERT_BATCH_SIZE for efficiency.
    Falls back to neutral (0.0) if inference fails for a headline.
    """
    if not headlines:
        return []

    finbert = get_finbert()
    texts   = [h["headline"] for h in headlines]
    scored  = []

    for i in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch_texts = texts[i:i + FINBERT_BATCH_SIZE]
        batch_meta  = headlines[i:i + FINBERT_BATCH_SIZE]

        try:
            results = finbert(batch_texts)
            for meta, result in zip(batch_meta, results):
                label      = result["label"].lower()
                confidence = result["score"]
                direction  = FINBERT_SCORE_MAP.get(label, 0.0)
                score      = direction * confidence

                scored.append({
                    **meta,
                    "sentiment":  label,
                    "score":      round(score, 4),
                    "confidence": round(confidence, 4),
                })

        except Exception as e:
            print(f"    FinBERT batch error: {e}")
            for meta in batch_meta:
                scored.append({
                    **meta,
                    "sentiment":  "neutral",
                    "score":      0.0,
                    "confidence": 0.0,
                })

    return scored


# ── GDELT LAYER FETCHING ──────────────────────────────────────────────

def fetch_gdelt_layer(queries: list[str], start: str, end: str,
                      layer_name: str) -> pd.Series:
    """
    Fetch and score a single GDELT sentiment layer for a date range.

    Strategy per query:
        1. Try artlist mode → get individual headlines
        2. Score headlines with FinBERT → continuous sentiment scores
        3. If artlist empty → fall back to timelinetone for that query
        4. Average scores across all queries in this layer

    This hybrid approach maximizes coverage:
        - FinBERT-scored artlist headlines where available (better quality)
        - GDELT pre-computed tone as fallback (reliable coverage)

    Args:
        queries    : list of query strings for this layer
        start      : date string 'YYYY-MM-DD'
        end        : date string 'YYYY-MM-DD'
        layer_name : 'company' or 'sector' — for logging only

    Returns Series indexed by date with averaged tone values.
    """
    all_series = []

    for query in queries:
        # Try artlist + FinBERT first
        headlines = fetch_gdelt_artlist(query, start, end)

        if headlines:
            scored  = score_with_finbert(headlines)
            records = [{"date": h["date"], "score": h["score"]} for h in scored]
            df      = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df      = df.groupby("date")["score"].mean()
            df.index = df.index.tz_localize(None)
            all_series.append(df)

        else:
            # Fall back to timelinetone
            tone_series = fetch_gdelt_timelinetone(query, start, end)
            if not tone_series.empty:
                # Normalize timelinetone to roughly same scale as FinBERT
                # GDELT tone typically -10 to +10, FinBERT is -1 to +1
                all_series.append(tone_series / 10.0)

        time.sleep(0.5)

    if not all_series:
        return pd.Series(dtype=float)

    combined = pd.concat(all_series, axis=1)
    return combined.mean(axis=1)


def compute_layer_features(series: pd.Series,
                            prefix: str) -> pd.DataFrame:
    """
    Compute derived features from a raw sentiment tone series.

    Called once per layer (company, sector) to produce consistent
    feature sets with the layer name as column prefix.

    Features:
        {prefix}_tone      : raw daily sentiment score
        {prefix}_tone_ma7  : 7-day smoothed — reduces daily noise
        {prefix}_tone_change: day-over-day change — sentiment momentum
        {prefix}_positive  : binary — tone > 0.1
        {prefix}_negative  : binary — tone < -0.1
    """
    df = pd.DataFrame({f"{prefix}_tone": series})
    df[f"{prefix}_tone_ma7"]    = df[f"{prefix}_tone"].rolling(7, min_periods=1).mean()
    df[f"{prefix}_tone_change"] = df[f"{prefix}_tone"].diff()
    df[f"{prefix}_positive"]    = (df[f"{prefix}_tone"] > 0.1).astype(int)
    df[f"{prefix}_negative"]    = (df[f"{prefix}_tone"] < -0.1).astype(int)
    return df


# ── GDELT ORCHESTRATION ───────────────────────────────────────────────

def fetch_gdelt_sentiment(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Orchestrate full GDELT sentiment pipeline for a ticker.

    Fetches headlines in monthly chunks — avoids GDELT timeouts
    on large date ranges and enables month-level caching so
    interrupted runs resume from the last completed month.

    Two sentiment layers computed independently:
        company layer : news directly about this company
        sector layer  : industry/supply-chain events affecting this company

    Final output columns:
        company_tone, company_tone_ma7, company_tone_change,
        company_positive, company_negative,
        sector_tone, sector_tone_ma7, sector_tone_change,
        sector_positive, sector_negative,
        gdelt_composite  ← weighted average of company + sector tone

    Caching:
        Saves monthly scores to data/{ticker}_gdelt_monthly_cache.csv
        Delete this file to force a full refetch.
        Final daily CSV saved to data/{ticker}_gdelt_daily.csv
    """
    queries    = get_gdelt_queries(ticker, company_name)
    train_end  = get_train_end(ticker)
    cache_path = data_path(f"{ticker}_gdelt_monthly_cache.csv")
    daily_path = data_path(f"{ticker}_gdelt_daily.csv")

    company_queries = queries.get("company", [])
    sector_queries  = queries.get("sector", [])

    # Load existing monthly cache
    cached_records = []
    cached_months  = set()

    if os.path.exists(cache_path):
        cached_df     = pd.read_csv(cache_path, parse_dates=["date"])
        cached_records = cached_df.to_dict("records")
        cached_months  = set(
            pd.to_datetime(cached_df["date"]).dt.to_period("M").astype(str)
        )
        print(f"  Cache: {len(cached_months)} months already scored")

    # Generate monthly chunks across training window
    start_dt = datetime.strptime(TRAIN_START, "%Y-%m-%d")
    end_dt   = datetime.strptime(train_end, "%Y-%m-%d")
    months   = []
    current  = start_dt

    while current <= end_dt:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        months.append((
            current.strftime("%Y-%m-%d"),
            min(next_month - timedelta(days=1), end_dt).strftime("%Y-%m-%d")
        ))
        current = next_month

    # Process each month — skip cached months
    new_records = []
    for month_start, month_end in months:
        period_key = pd.to_datetime(month_start).to_period("M").strftime("%Y-%m")
        if period_key in cached_months:
            continue

        print(f"    {period_key}...", end=" ", flush=True)

        company_series = fetch_gdelt_layer(
            company_queries, month_start, month_end, "company"
        )
        sector_series = fetch_gdelt_layer(
            sector_queries, month_start, month_end, "sector"
        )

        # Combine into records for this month
        all_dates = set()
        if not company_series.empty:
            all_dates.update(company_series.index.tolist())
        if not sector_series.empty:
            all_dates.update(sector_series.index.tolist())

        for date in all_dates:
            comp_tone   = float(company_series.get(date, np.nan))
            sector_tone = float(sector_series.get(date, np.nan))
            new_records.append({
                "date":         date,
                "company_tone": comp_tone,
                "sector_tone":  sector_tone,
            })

        print("done")

    # Merge cached + new records
    all_records = cached_records + new_records

    if not all_records:
        print(f"  No GDELT data retrieved for {ticker}")
        return pd.DataFrame()

    # Save updated monthly cache
    cache_df = pd.DataFrame(all_records)
    cache_df["date"] = pd.to_datetime(cache_df["date"])
    cache_df.to_csv(cache_path, index=False)

    # Build final daily DataFrame
    daily = cache_df.set_index("date").sort_index()
    daily.index = daily.index.tz_localize(None)
    daily = daily.groupby(daily.index).mean()

    # Compute derived features per layer
    company_features = compute_layer_features(
        daily["company_tone"].dropna(), "company"
    )
    sector_features = compute_layer_features(
        daily["sector_tone"].dropna(), "sector"
    )

    # Join layers
    result = company_features.join(sector_features, how="outer")

    # Weighted composite — filled with available layer if one is missing
    result["gdelt_composite"] = (
        result["company_tone"].fillna(0) * GDELT_COMPANY_WEIGHT +
        result["sector_tone"].fillna(0) * GDELT_SECTOR_WEIGHT
    )

    result.to_csv(daily_path)
    print(f"  Saved {len(result)} rows to {daily_path}")
    return result


# ── COMPANY NAME ──────────────────────────────────────────────────────

def get_company_name(ticker: str) -> str:
    """Fetch full company name from yfinance — falls back to ticker."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


# ── SAVE ──────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to data/ directory as CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = data_path(filename)
    df.to_csv(path)
    print(f"  Saved {filename} ({len(df)} rows)")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────

def run_data_pipeline(tickers: list = None) -> None:
    """
    Run the full data fetching pipeline for all configured tickers.

    Delete data/{ticker}_gdelt_monthly_cache.csv to force
    a full GDELT refetch for a specific ticker.
    """
    if tickers is None:
        tickers = PRETRAINED_TICKERS

    print(f"\nStarting data pipeline for: {tickers}")
    print(f"Training window: {TRAIN_START} → {get_train_end('default')}\n")

    # Stage 1 — OHLCV
    print("=" * 50)
    print("Stage 1: OHLCV Data")
    print("=" * 50)
    for ticker in tickers:
        try:
            df = fetch_market_data(ticker)
            save_data(df, f"{ticker}_raw.csv")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    # Stage 2 — Macro
    print("\n" + "=" * 50)
    print("Stage 2: Macro Data (VIX, TNX)")
    print("=" * 50)
    try:
        macro = fetch_macro_data()
        save_data(macro, "macro.csv")
    except Exception as e:
        print(f"  ERROR macro: {e}")

    # Stage 3 — Earnings
    print("\n" + "=" * 50)
    print("Stage 3: Earnings Dates")
    print("=" * 50)
    for ticker in tickers:
        try:
            earnings = fetch_earnings_dates(ticker)
            if not earnings.empty:
                save_data(earnings, f"{ticker}_earnings.csv")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    # Stage 4 — Google Trends
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
            print(f"  ERROR {ticker}: {e}")

    # Stage 5 — GDELT + FinBERT
    print("\n" + "=" * 50)
    print("Stage 5: GDELT Sentiment (FinBERT-scored)")
    print("=" * 50)
    print("Caching enabled — delete {ticker}_gdelt_monthly_cache.csv to refetch\n")
    for ticker in tickers:
        try:
            company_name = get_company_name(ticker)
            print(f"Processing {ticker} ({company_name})...")
            fetch_gdelt_sentiment(ticker, company_name)
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("Pipeline complete. Files saved:")
    print("=" * 50)
    for f in sorted(os.listdir(DATA_DIR)):
        size = os.path.getsize(data_path(f)) / 1024
        print(f"  {f:<45} {size:>8.1f} KB")


if __name__ == "__main__":
    # Delete old GDELT daily CSVs — new pipeline produces different columns
    for ticker in PRETRAINED_TICKERS:
        old_daily = data_path(f"{ticker}_gdelt_daily.csv")
        old_cache = data_path(f"{ticker}_gdelt_monthly_cache.csv")
        old_raw   = data_path(f"{ticker}_gdelt_raw.csv")
        for path in [old_daily, old_cache, old_raw]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleared old cache: {path}")

    run_data_pipeline()