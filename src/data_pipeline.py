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
from io import StringIO
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

def is_gdelt_cache_complete(ticker: str) -> bool:
    """
    Return True if GDELT daily CSV exists and covers the full
    training window within a 35-day tolerance.

    35-day tolerance handles month boundary gaps — the last month
    may not have a full 31 days of data but the cache is still valid.
    """
    daily_path = data_path(f"{ticker}_gdelt_daily.csv")
    if not os.path.exists(daily_path):
        return False

    try:
        df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        if df.empty:
            return False

        train_end = pd.to_datetime(get_train_end(ticker))
        last_date = pd.to_datetime(df.index).max()
        gap_days  = (train_end - last_date).days
        return gap_days <= 35

    except Exception:
        return False


def fetch_gdelt_sentiment(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Orchestrate GDELT sentiment pipeline with three-level caching.

    Level 1 — Daily CSV completeness check:
        If daily CSV exists and covers full training window,
        return immediately — no API calls, no FinBERT loading.
        This is the common case on repeated pipeline runs.

    Level 2 — Monthly chunk cache:
        Tracks which months have already been fetched and scored.
        Skips completed months — only processes new months.
        FinBERT loads lazily — only initializes if scoring is needed.

    Level 3 — Headline hash cache:
        Within a month, if the exact same headlines were already
        scored in a previous run, returns cached scores without
        calling FinBERT at all.

    Delete data/{ticker}_gdelt_daily.csv to force full refetch.
    Delete data/{ticker}_gdelt_monthly_cache.csv to force re-scoring.
    """
    import json
    import hashlib

    daily_path  = data_path(f"{ticker}_gdelt_daily.csv")
    cache_path  = data_path(f"{ticker}_gdelt_monthly_cache.csv")
    hash_path   = data_path(f"{ticker}_score_hashes.json")

    # ── Level 1: Full cache check ─────────────────────────────────────
    if is_gdelt_cache_complete(ticker):
        cached = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        print(f"  Cache complete — skipping FinBERT ({len(cached)} rows)")
        return cached

    # ── Load monthly cache ────────────────────────────────────────────
    cached_records = []
    cached_months  = set()

    if os.path.exists(cache_path):
        cached_df      = pd.read_csv(cache_path, parse_dates=["date"])
        cached_records = cached_df.to_dict("records")
        cached_months  = set(
            pd.to_datetime(cached_df["date"]).dt.to_period("M").astype(str)
        )

    # ── Load hash cache ───────────────────────────────────────────────
    hash_cache = {}
    if os.path.exists(hash_path):
        with open(hash_path) as f:
            try:
                hash_cache = json.load(f)
            except Exception:
                hash_cache = {}

    # ── Generate monthly chunks ───────────────────────────────────────
    queries   = get_gdelt_queries(ticker, company_name)
    train_end = get_train_end(ticker)

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

    # ── Level 2 + 3: Process uncached months ─────────────────────────
    new_records   = []
    months_scored = 0

    for month_start, month_end in months:
        period_key = pd.to_datetime(month_start).to_period("M").strftime("%Y-%m")

        if period_key in cached_months:
            continue   # Level 2 — skip completed month

        print(f"    {period_key}...", end=" ", flush=True)

        company_series = pd.Series(dtype=float)
        sector_series  = pd.Series(dtype=float)

        for layer_name, layer_queries in [
            ("company", queries.get("company", [])),
            ("sector",  queries.get("sector", []))
        ]:
            layer_scores = []

            for query in layer_queries:
                headlines = fetch_gdelt_artlist(query, month_start, month_end)

                if headlines:
                    # Level 3 — check headline hash cache
                    content    = "|".join(sorted(h["headline"] for h in headlines))
                    hl_hash    = hashlib.md5(content.encode()).hexdigest()
                    cache_key  = f"{ticker}_{period_key}_{layer_name}_{hl_hash}"

                    if cache_key in hash_cache:
                        scored = hash_cache[cache_key]
                    else:
                        # FinBERT loads lazily here — only when actually needed
                        scored = score_with_finbert(headlines)
                        hash_cache[cache_key] = scored

                    records_df = pd.DataFrame([
                        {"date": h["date"], "score": h["score"]}
                        for h in scored
                    ])
                    if not records_df.empty:
                        records_df["date"] = pd.to_datetime(records_df["date"])
                        series = records_df.groupby("date")["score"].mean()
                        series.index = series.index.tz_localize(None)
                        layer_scores.append(series)

                else:
                    # Fallback to timelinetone
                    tone = fetch_gdelt_timelinetone(query, month_start, month_end)
                    if not tone.empty:
                        layer_scores.append(tone / 10.0)

                time.sleep(0.5)

            if layer_scores:
                combined = pd.concat(layer_scores, axis=1).mean(axis=1)
                if layer_name == "company":
                    company_series = combined
                else:
                    sector_series  = combined

        # Collect dates from both layers
        all_dates = set()
        if not company_series.empty:
            all_dates.update(company_series.index.tolist())
        if not sector_series.empty:
            all_dates.update(sector_series.index.tolist())

        for date in all_dates:
            new_records.append({
                "date":         date,
                "company_tone": float(company_series.get(date, np.nan)),
                "sector_tone":  float(sector_series.get(date, np.nan)),
            })

        months_scored += 1
        print("done")

    # ── Persist hash cache ────────────────────────────────────────────
    if months_scored > 0:
        with open(hash_path, "w") as f:
            json.dump(hash_cache, f)

    # ── Build final daily CSV ─────────────────────────────────────────
    all_records = cached_records + new_records

    if not all_records:
        print(f"  No GDELT data for {ticker}")
        return pd.DataFrame()

    cache_df = pd.DataFrame(all_records)
    cache_df["date"] = pd.to_datetime(cache_df["date"])
    cache_df.to_csv(cache_path, index=False)

    daily = cache_df.set_index("date").sort_index()
    daily.index = daily.index.tz_localize(None)
    daily = daily.groupby(daily.index).mean()

    company_features = compute_layer_features(
        daily["company_tone"].dropna(), "company"
    )
    sector_features = compute_layer_features(
        daily["sector_tone"].dropna(), "sector"
    )

    result = company_features.join(sector_features, how="outer")
    result["gdelt_composite"] = (
        result["company_tone"].fillna(0) * GDELT_COMPANY_WEIGHT +
        result["sector_tone"].fillna(0) * GDELT_SECTOR_WEIGHT
    )

    result.to_csv(daily_path)
    print(f"  Saved {len(result)} rows → {daily_path}")
    return result

# ── INSIDER TRANSACTIONS ──────────────────────────────────────────────

def fetch_insider_raw(ticker: str) -> pd.DataFrame:
    """
    Fetch raw insider transaction data from OpenInsider via CSV export.

    OpenInsider exposes a screener endpoint that returns CSV directly.
    We fetch all transactions since TRAIN_START — the date range
    parameter (fd) is specified in days from today.

    Filters applied at fetch time:
        xp=1 : exclude option exercises — not discretionary buys
        xs=1 : exclude automatic 10b5-1 plan sales — pre-scheduled,
                not opportunistic, carry no information signal
        xn=1 : exclude non-open-market transactions (gifts, grants)

    Returns raw DataFrame with all transactions or empty DataFrame
    on failure.
    """
    from config import OPENINSIDER_URL, TRAIN_START, INSIDER_MIN_VALUE

    # Calculate days since TRAIN_START for the fd parameter
    days_back = (datetime.now() - datetime.strptime(TRAIN_START, "%Y-%m-%d")).days + 30

    params = {
        "s":      ticker,
        "fd":     days_back,       # from date in days
        "xp":     1,               # exclude option exercises
        "xs":     1,               # exclude 10b5-1 automatic sales
        "xn":     1,               # exclude non-open-market
        "action": "1",
        "cnt":    "500",           # max records
    }

    try:
        # OpenInsider returns CSV when you append &action=1 to screener URL
        csv_url = f"{OPENINSIDER_URL}?s={ticker}&fd={days_back}&xp=1&xs=1&xn=1&cnt=500&action=1"
        response = requests.get(
            csv_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if not response.text.strip():
            print(f"  Empty response from OpenInsider for {ticker}")
            return pd.DataFrame()

        # OpenInsider returns HTML — parse the table
        tables = pd.read_html(StringIO(response.text))

        if not tables:
            print(f"  No tables found in OpenInsider response for {ticker}")
            return pd.DataFrame()

        # The main transaction table is always the largest
        df = max(tables, key=len)

        if df.empty:
            return pd.DataFrame()

        # Standardize column names — OpenInsider columns vary slightly
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

        print(f"  Raw insider rows: {len(df)}")
        return df

    except Exception as e:
        print(f"  OpenInsider fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def parse_insider_transactions(df_raw: pd.DataFrame,
                               ticker: str) -> pd.DataFrame:
    """
    Parse and clean raw OpenInsider HTML table into structured DataFrame.

    OpenInsider's HTML table has inconsistent column naming across
    different tickers and time periods. This function handles the
    mapping robustly using partial string matching rather than
    exact column names.

    Filters applied:
        - Only open-market purchases (transaction type 'P')
        - Value >= INSIDER_MIN_VALUE
        - Valid trade dates within training window

    Returns cleaned DataFrame with standardized columns:
        trade_date  : datetime of the transaction
        insider     : name of the insider
        title       : their role/title
        is_buy      : 1 for purchase, 0 for sale
        value       : dollar value of transaction
        is_exec     : 1 if CEO or CFO
    """
    from config import INSIDER_MIN_VALUE, INSIDER_EXEC_TITLES, TRAIN_START, INSIDER_MIN_VALUE

    if df_raw.empty:
        return pd.DataFrame()

    # Find columns by partial name matching — robust to naming variations
    def find_col(df, keywords):
        for col in df.columns:
            if any(k in col for k in keywords):
                return col
        return None

    date_col  = find_col(df_raw, ["trade", "date"])
    title_col = find_col(df_raw, ["title", "relation", "role"])
    type_col  = find_col(df_raw, ["type", "transaction"])
    value_col = find_col(df_raw, ["value", "$value", "amt"])
    name_col  = find_col(df_raw, ["name", "insider", "filer"])

    # If critical columns missing, return empty
    if not date_col or not type_col:
        print(f"  Could not identify required columns in {ticker} insider data")
        print(f"  Available: {df_raw.columns.tolist()}")
        return pd.DataFrame()

    records = []
    for _, row in df_raw.iterrows():
        try:
            # Parse trade date
            raw_date = str(row.get(date_col, ""))
            try:
                trade_date = pd.to_datetime(raw_date[:10])
            except Exception:
                continue

            # Filter to training window
            if trade_date < pd.to_datetime(TRAIN_START):
                continue

            # Transaction type — only purchases
            txn_type = str(row.get(type_col, "")).strip().upper()
            is_buy   = 1 if txn_type in ["P", "PURCHASE", "BUY"] else 0

            # Dollar value — clean currency formatting
            raw_value = str(row.get(value_col, "0")) if value_col else "0"
            raw_value = raw_value.replace("$", "").replace(",", "").replace("+", "").strip()
            try:
                value = abs(float(raw_value))
            except Exception:
                value = 0.0

            # Filter minimum value
            if value < INSIDER_MIN_VALUE:
                continue

            # Insider title
            title = str(row.get(title_col, "")).lower() if title_col else ""
            name  = str(row.get(name_col, "")) if name_col else ""

            # Executive flag — CEO/CFO are highest conviction
            is_exec = int(any(t in title for t in INSIDER_EXEC_TITLES))

            records.append({
                "trade_date": trade_date,
                "insider":    name,
                "title":      title,
                "is_buy":     is_buy,
                "value":      value,
                "is_exec":    is_exec,
            })

        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.tz_localize(None)
    result = result.sort_values("trade_date")
    print(f"  Parsed {len(result)} valid transactions "
          f"({result['is_buy'].sum()} buys, "
          f"{(1-result['is_buy']).sum()} sells)")
    return result


def compute_insider_features(transactions: pd.DataFrame,
                              price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute rolling insider transaction features aligned to trading days.

    For each trading day in price_index, looks back INSIDER_WINDOW_DAYS
    and computes aggregate features from all transactions in that window.

    Features:
        insider_buy_count_30d  : number of open-market purchases
        insider_sell_count_30d : number of open-market sales
        insider_net_30d        : buy_count - sell_count
        insider_buy_value_30d  : log(1 + total buy value) — log-scaled
        insider_sell_value_30d : log(1 + total sell value) — log-scaled
        insider_cluster_buy    : 1 if 2+ insiders bought in window
        exec_bought_30d        : 1 if CEO or CFO bought in window

    Log-scaling value features prevents large transactions from
    dominating the feature — a $50M CEO purchase vs $100k director
    purchase becomes log(50M) vs log(100k) = 17.7 vs 11.5,
    a manageable ratio rather than 500x.
    """
    from config import INSIDER_WINDOW_DAYS

    if transactions.empty:
        # Return zeros — model treats no insider data as neutral
        result = pd.DataFrame(index=price_index)
        for col in ["insider_buy_count_30d", "insider_sell_count_30d",
                    "insider_net_30d", "insider_buy_value_30d",
                    "insider_sell_value_30d", "insider_cluster_buy",
                    "exec_bought_30d"]:
            result[col] = 0
        return result

    records = []
    for date in price_index:
        window_start = date - timedelta(days=INSIDER_WINDOW_DAYS)

        # All transactions in the rolling window
        window = transactions[
            (transactions["trade_date"] >= window_start) &
            (transactions["trade_date"] <= date)
        ]

        buys  = window[window["is_buy"] == 1]
        sells = window[window["is_buy"] == 0]

        # Cluster buy — multiple insiders buying signals conviction
        unique_buyers = buys["insider"].nunique() if not buys.empty else 0

        records.append({
            "date":                    date,
            "insider_buy_count_30d":   len(buys),
            "insider_sell_count_30d":  len(sells),
            "insider_net_30d":         len(buys) - len(sells),
            "insider_buy_value_30d":   np.log1p(buys["value"].sum()),
            "insider_sell_value_30d":  np.log1p(sells["value"].sum()),
            "insider_cluster_buy":     int(unique_buyers >= 2),
            "exec_bought_30d":         int(buys["is_exec"].sum() > 0)
                                       if not buys.empty else 0,
        })

    result = pd.DataFrame(records).set_index("date")
    result.index = pd.to_datetime(result.index).tz_localize(None)
    return result


def fetch_insider_transactions(ticker: str) -> pd.DataFrame:
    """
    Orchestrate full insider transaction pipeline for a ticker.

    Stages:
        1. Fetch raw HTML table from OpenInsider
        2. Parse and clean into structured transactions
        3. Compute rolling features aligned to trading day index

    Caches raw transactions to data/{ticker}_insider_raw.csv.
    Saves daily features to data/{ticker}_insider_daily.csv.

    Falls back gracefully — returns zero-filled features if
    OpenInsider is unavailable. ml_forecasting.py merges these
    features the same way regardless of whether real data or zeros.
    """
    cache_path = data_path(f"{ticker}_insider_raw.csv")
    daily_path = data_path(f"{ticker}_insider_daily.csv")

    # Load raw transactions from cache if available
    if os.path.exists(cache_path):
        print(f"  Loading insider transactions from cache...")
        transactions = pd.read_csv(cache_path, parse_dates=["trade_date"])
        transactions["trade_date"] = pd.to_datetime(
            transactions["trade_date"]
        ).dt.tz_localize(None)
    else:
        # Fetch from OpenInsider
        df_raw       = fetch_insider_raw(ticker)
        transactions = parse_insider_transactions(df_raw, ticker)

        if not transactions.empty:
            transactions.to_csv(cache_path, index=False)
            print(f"  Cached {len(transactions)} transactions")

    # Load price index to align features to trading days
    raw_path = data_path(f"{ticker}_raw.csv")
    if not os.path.exists(raw_path):
        print(f"  Warning: {ticker}_raw.csv not found — cannot align insider features")
        return pd.DataFrame()

    price_df    = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    price_index = pd.to_datetime(price_df.index).tz_localize(None)

    # Compute rolling features
    daily = compute_insider_features(transactions, price_index)

    daily.to_csv(daily_path)
    print(f"  Saved insider features to {daily_path} ({len(daily)} rows)")
    return daily

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
    
    # Stage 6 — Insider Transactions
    print("\n" + "=" * 50)
    print("Stage 6: Insider Transactions (OpenInsider)")
    print("=" * 50)
    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")
            fetch_insider_transactions(ticker)
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