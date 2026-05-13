# src/data_pipeline.py
#
# Data fetching layer for the quant research assistant.
#
# Stages:
#   1 — OHLCV price data per ticker
#   2 — Macro context (VIX, TNX) — shared across tickers
#   3 — Earnings announcement dates per ticker
#   4 — Google Trends attention signal per ticker
#   5 — GDELT sentiment scored by FinBERT per ticker
#   6 — Insider transactions via SEC EDGAR per ticker
#
# Caching:
#   Every stage caches to disk — reruns are fast.
#   Stage 5 (GDELT): two-level cache — daily CSV + monthly chunks.
#     FinBERT only loads when genuinely uncached months exist.
#   Stage 6 (Insider): caches raw transactions + daily features.
#
# Run once before training:
#   python src/data_pipeline.py
#
# To force a refetch:
#   Trends:  delete data/{ticker}_trends.csv
#   GDELT:   delete data/{ticker}_gdelt_daily.csv
#            delete data/{ticker}_gdelt_monthly_cache.csv
#   Insider: delete data/{ticker}_insider_daily.csv
#            delete data/{ticker}_insider_raw.csv

import os
import sys
import re
import time
import requests
import warnings
import xml.etree.ElementTree as ET
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
    INSIDER_EXEC_TITLES,
    INSIDER_MIN_VALUE,
    INSIDER_WINDOW_DAYS,
    DATA_DIR,
    data_path,
    get_train_end,
    get_trends_query,
    get_gdelt_queries,
    is_etf,
)

warnings.filterwarnings("ignore")


# ── FINBERT ───────────────────────────────────────────────────────────

_finbert_pipeline = None


def get_finbert():
    """
    Lazy-load FinBERT pipeline — only initializes on first call.

    Downloads ~440MB on first use, cached permanently by HuggingFace.
    Never called if all GDELT monthly caches are complete (Level 1 hit).
    CPU inference only — set device=0 for GPU if available.
    """
    global _finbert_pipeline
    if _finbert_pipeline is None:
        print("  Loading FinBERT model...")
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            task="text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            max_length=FINBERT_MAX_LENGTH,
            truncation=True,
            device=-1,
        )
        print("  FinBERT loaded")
    return _finbert_pipeline


# ── STAGE 1: OHLCV ───────────────────────────────────────────────────

def fetch_market_data(ticker: str) -> pd.DataFrame:
    """
    Fetch OHLCV within the fixed training window.

    Uses TRAIN_START and per-ticker TRAIN_END from config —
    reproducible regardless of when the script is run.
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


# ── STAGE 2: MACRO ────────────────────────────────────────────────────

def fetch_macro_data() -> pd.DataFrame:
    """
    Fetch VIX and TNX for the full training window.

    Ticker-independent — fetched once, merged into every feature matrix.
    Derived features: vix_change, tnx_change, vix_ma20.
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


# ── STAGE 3: EARNINGS ─────────────────────────────────────────────────

def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Fetch earnings dates within training window.
    Skipped automatically for ETFs.
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


# ── STAGE 4: GOOGLE TRENDS ────────────────────────────────────────────

def fetch_trends_data(ticker: str, company_name: str) -> pd.DataFrame:
    """
    Fetch Google Trends attention signal — weekly resolution.

    Attention signal not sentiment — measures search interest (0-100).
    Standardized to weekly resolution for both training and inference.

    Features:
        trends_score   : raw weekly index (0-100)
        trends_change  : week-over-week pct change
        trends_zscore  : deviation from 52-week rolling baseline
        trends_spike   : binary — zscore > 2.0
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
        df["trends_zscore"] = (
            df["trends_score"] - rolling_mean
        ) / rolling_std.replace(0, 1)
        df["trends_spike"]  = (df["trends_zscore"] > 2.0).astype(int)

        df.dropna(inplace=True)
        print(f"  Retrieved {len(df)} weekly rows")
        return df

    except Exception as e:
        print(f"  Trends fetch failed: {e}")
        return pd.DataFrame()


# ── STAGE 5: GDELT SENTIMENT ──────────────────────────────────────────

def fetch_gdelt_artlist(query: str, start: str, end: str) -> list:
    """
    Fetch individual headlines from GDELT artlist mode.

    Returns list of {headline, date} dicts.
    Returns empty list on failure — caller falls back to timelinetone.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

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

        articles = response.json().get("articles", [])
        results  = []
        for article in articles:
            raw_date = article.get("seendate", "")
            try:
                date = datetime.strptime(
                    raw_date[:8], "%Y%m%d"
                ).strftime("%Y-%m-%d")
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

    Fallback when artlist returns empty for a period.
    Returns Series indexed by date with pre-computed tone values.
    Normalized to FinBERT scale (/10) before returning.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

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

        timeline = response.json().get("timeline", [])
        records  = []
        for series in timeline:
            for point in series.get("data", []):
                raw_date = point.get("date", "")
                tone     = point.get("value", 0.0)
                try:
                    date = (
                        datetime.strptime(
                            raw_date[:8], "%Y%m%d"
                        ).strftime("%Y-%m-%d")
                        if "T" in raw_date else raw_date[:10]
                    )
                    records.append({"date": date, "tone": float(tone)})
                except Exception:
                    continue

        if not records:
            return pd.Series(dtype=float)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df.index = df.index.tz_localize(None)
        # Normalize GDELT tone (-10 to +10) to FinBERT scale (-1 to +1)
        return df.groupby(df.index)["tone"].mean() / 10.0

    except Exception:
        return pd.Series(dtype=float)


def score_with_finbert(headlines: list) -> list:
    """
    Score headlines using FinBERT locally.

    Converts label + confidence to continuous score:
        positive × confidence → near +1.0
        negative × confidence → near -1.0
        neutral               → 0.0

    Falls back to neutral on batch failure.
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
                label     = result["label"].lower()
                direction = FINBERT_SCORE_MAP.get(label, 0.0)
                score     = direction * result["score"]
                scored.append({
                    **meta,
                    "sentiment":  label,
                    "score":      round(score, 4),
                    "confidence": round(result["score"], 4),
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


def fetch_gdelt_layer(queries: list, start: str, end: str) -> pd.Series:
    """
    Fetch and score one GDELT sentiment layer for a date range.

    Per query:
        1. artlist mode → FinBERT scoring (financially aware)
        2. artlist empty → timelinetone fallback (reliable coverage)

    Averages across all queries in the layer for robustness.
    """
    all_series = []

    for query in queries:
        headlines = fetch_gdelt_artlist(query, start, end)

        if headlines:
            scored  = score_with_finbert(headlines)
            records = [{"date": h["date"], "score": h["score"]}
                       for h in scored]
            df      = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            series  = df.groupby("date")["score"].mean()
            series.index = series.index.tz_localize(None)
            all_series.append(series)
        else:
            tone = fetch_gdelt_timelinetone(query, start, end)
            if not tone.empty:
                all_series.append(tone)

        time.sleep(0.5)

    if not all_series:
        return pd.Series(dtype=float)

    return pd.concat(all_series, axis=1).mean(axis=1)


def compute_layer_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Compute derived sentiment features from a raw tone series.

    Features:
        {prefix}_tone        : raw daily sentiment score
        {prefix}_tone_ma7    : 7-day smoothed — reduces noise
        {prefix}_tone_change : day-over-day momentum
        {prefix}_positive    : binary — tone > 0.1
        {prefix}_negative    : binary — tone < -0.1
    """
    df = pd.DataFrame({f"{prefix}_tone": series})
    df[f"{prefix}_tone_ma7"]    = df[f"{prefix}_tone"].rolling(
        7, min_periods=1
    ).mean()
    df[f"{prefix}_tone_change"] = df[f"{prefix}_tone"].diff()
    df[f"{prefix}_positive"]    = (df[f"{prefix}_tone"] > 0.1).astype(int)
    df[f"{prefix}_negative"]    = (df[f"{prefix}_tone"] < -0.1).astype(int)
    return df


def is_gdelt_cache_complete(ticker: str) -> bool:
    """
    True if daily CSV exists and covers the full training window
    within a 35-day tolerance for month boundary gaps.
    """
    daily_path = data_path(f"{ticker}_gdelt_daily.csv")
    if not os.path.exists(daily_path):
        return False
    try:
        df        = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        if df.empty:
            return False
        train_end = pd.to_datetime(get_train_end(ticker))
        last_date = pd.to_datetime(df.index).max()
        return (train_end - last_date).days <= 35
    except Exception:
        return False


def _build_daily_from_records(records: list, daily_path: str) -> pd.DataFrame:
    """
    Build and save daily sentiment CSV from monthly cache records.
    Called when monthly cache is complete but daily CSV needs rebuilding.
    """
    if not records:
        return pd.DataFrame()

    cache_df = pd.DataFrame(records)
    cache_df["date"] = pd.to_datetime(cache_df["date"])

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
        result["sector_tone"].fillna(0)  * GDELT_SECTOR_WEIGHT
    )
    result.to_csv(daily_path)
    return result


def fetch_gdelt_sentiment(ticker: str, company_name: str) -> pd.DataFrame:
    """
    GDELT sentiment pipeline with two-level caching.

    Level 1 — daily CSV complete → return instantly, no FinBERT loaded.
    Level 2 — monthly cache → skip completed months.

    FinBERT only loads when genuinely uncached months exist.

    Cache management:
        Delete {ticker}_gdelt_daily.csv          → forces daily CSV rebuild
        Delete {ticker}_gdelt_monthly_cache.csv  → forces full rescore
    """
    daily_path = data_path(f"{ticker}_gdelt_daily.csv")
    cache_path = data_path(f"{ticker}_gdelt_monthly_cache.csv")

    # ── Level 1: Full cache check ─────────────────────────────────────
    if is_gdelt_cache_complete(ticker):
        cached = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        print(f"  Cache complete — skipping ({len(cached)} rows)")
        return cached

    # ── Level 2: Monthly cache ────────────────────────────────────────
    cached_records = []
    cached_months  = set()

    if os.path.exists(cache_path):
        cached_df      = pd.read_csv(cache_path, parse_dates=["date"])
        cached_records = cached_df.to_dict("records")
        cached_months  = set(
            pd.to_datetime(cached_df["date"]).dt.to_period("M").astype(str)
        )

    # Generate monthly chunks across training window
    train_end = get_train_end(ticker)
    start_dt  = datetime.strptime(TRAIN_START, "%Y-%m-%d")
    end_dt    = datetime.strptime(train_end, "%Y-%m-%d")
    months    = []
    current   = start_dt

    while current <= end_dt:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        months.append((
            current.strftime("%Y-%m-%d"),
            min(next_month - timedelta(days=1), end_dt).strftime("%Y-%m-%d")
        ))
        current = next_month

    months_needed = [
        (s, e) for s, e in months
        if pd.to_datetime(s).to_period("M").strftime("%Y-%m")
        not in cached_months
    ]

    # If all months cached, rebuild daily CSV without touching FinBERT
    if not months_needed:
        print("  All months cached — rebuilding daily CSV...")
        return _build_daily_from_records(cached_records, daily_path)

    # FinBERT only loads here — confirmed work needed
    print(f"  {len(months_needed)} months need scoring...")
    queries     = get_gdelt_queries(ticker, company_name)
    new_records = []

    for month_start, month_end in months_needed:
        period_key = (
            pd.to_datetime(month_start).to_period("M").strftime("%Y-%m")
        )
        print(f"    {period_key}...", end=" ", flush=True)

        company_series = fetch_gdelt_layer(
            queries.get("company", []), month_start, month_end
        )
        sector_series = fetch_gdelt_layer(
            queries.get("sector", []),  month_start, month_end
        )

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

        print("done")

    all_records      = cached_records + new_records
    cache_df         = pd.DataFrame(all_records)
    cache_df["date"] = pd.to_datetime(cache_df["date"])
    cache_df.to_csv(cache_path, index=False)

    return _build_daily_from_records(all_records, daily_path)


# ── STAGE 6: INSIDER TRANSACTIONS ────────────────────────────────────

def get_cik_for_ticker(ticker: str) -> str:
    """
    Get SEC CIK for a ticker via EDGAR company tickers JSON.
    CIK is required to fetch Form 4 filings from EDGAR.
    """
    try:
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "QuantResearch contact@example.com"},
            timeout=15,
        )
        for entry in response.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        print(f"  CIK not found for {ticker}")
        return None
    except Exception as e:
        print(f"  CIK lookup failed: {e}")
        return None


def fetch_insider_raw(ticker: str) -> pd.DataFrame:
    """
    Fetch and parse Form 4 insider transactions from SEC EDGAR.

    Uses official EDGAR submissions API + Form 4 XML parsing.
    No HTML scraping — authoritative, structured source.

    Only processes transaction codes:
        P : open-market purchase → is_buy=1  (strong signal)
        S : open-market sale     → is_buy=0  (weak signal)
        A, M, F, D → ignored    (non-discretionary)

    Returns DataFrame: trade_date, insider, title, is_buy, value, is_exec
    """
    headers = {"User-Agent": "QuantResearch contact@example.com"}

    cik = get_cik_for_ticker(ticker)
    if not cik:
        return pd.DataFrame()

    try:
        url      = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=headers, timeout=20)
        data     = response.json()
    except Exception as e:
        print(f"  EDGAR API failed: {e}")
        return pd.DataFrame()

    filings    = data.get("filings", {}).get("recent", {})
    forms      = filings.get("form", [])
    dates      = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])

    train_start = pd.to_datetime(TRAIN_START)
    train_end   = pd.to_datetime(get_train_end(ticker))

    form4_list = [
        (date, acc)
        for form, date, acc in zip(forms, dates, accessions)
        if form in ["4", "4/A"]
        and train_start <= pd.to_datetime(date) <= train_end
    ]

    if not form4_list:
        print(f"  No Form 4 filings in training window for {ticker}")
        return pd.DataFrame()

    print(f"  Found {len(form4_list)} Form 4 filings — parsing...")

    records = []
    for filing_date, acc_num in form4_list[:200]:
        try:
            acc_clean = acc_num.replace("-", "")
            acc_url   = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/"
            )
            r = requests.get(acc_url, headers=headers, timeout=10)

            xml_files = re.findall(
                r'href="([^"]+\.xml)"', r.text, re.IGNORECASE
            )
            if not xml_files:
                continue

            xml_url = (
                f"https://www.sec.gov{xml_files[0]}"
                if xml_files[0].startswith("/") else xml_files[0]
            )
            xml_r = requests.get(xml_url, headers=headers, timeout=10)
            if not xml_r.text.strip():
                continue

            root     = ET.fromstring(xml_r.content)
            reporter = root.find(".//reportingOwner")
            name     = ""
            title    = ""
            is_exec  = 0

            if reporter is not None:
                name_el  = reporter.find(".//rptOwnerName")
                title_el = reporter.find(".//officerTitle")
                name     = (name_el.text.strip()
                            if name_el is not None and name_el.text else "")
                title    = (title_el.text.strip().lower()
                            if title_el is not None and title_el.text else "")
                is_exec  = int(
                    any(t in title for t in INSIDER_EXEC_TITLES)
                )

            for txn in root.findall(".//nonDerivativeTransaction"):
                code_el   = txn.find(".//transactionCode")
                date_el   = txn.find(".//transactionDate/value")
                shares_el = txn.find(".//transactionShares/value")
                price_el  = txn.find(".//transactionPricePerShare/value")

                if code_el is None or date_el is None:
                    continue

                code = code_el.text.strip().upper() if code_el.text else ""
                if code not in ["P", "S"]:
                    continue

                try:
                    trade_date = pd.to_datetime(date_el.text.strip())
                except Exception:
                    trade_date = pd.to_datetime(filing_date)

                if not (train_start <= trade_date <= train_end):
                    continue

                try:
                    shares = (float(shares_el.text.strip())
                              if shares_el is not None
                              and shares_el.text else 0)
                    price  = (float(price_el.text.strip())
                              if price_el is not None
                              and price_el.text else 0)
                    value  = shares * price
                except Exception:
                    value = 0.0

                if value < INSIDER_MIN_VALUE:
                    continue

                records.append({
                    "trade_date": trade_date,
                    "insider":    name,
                    "title":      title,
                    "is_buy":     1 if code == "P" else 0,
                    "value":      value,
                    "is_exec":    is_exec,
                })

            time.sleep(0.1)

        except Exception:
            continue

    if not records:
        print(f"  No valid transactions parsed for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.tz_localize(None)
    df = df.sort_values("trade_date").reset_index(drop=True)

    buys  = df["is_buy"].sum()
    sells = len(df) - buys
    print(f"  Parsed {len(df)} transactions ({buys} buys, {sells} sells)")
    return df


def compute_insider_features(transactions: pd.DataFrame,
                              price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Rolling 30-day insider features aligned to trading days.

    Features:
        insider_buy_count_30d   : open-market purchases in window
        insider_sell_count_30d  : open-market sales in window
        insider_net_30d         : buy_count - sell_count
        insider_buy_value_30d   : log(1 + total buy $)
        insider_sell_value_30d  : log(1 + total sell $)
        insider_cluster_buy     : 1 if 2+ unique insiders bought
        exec_bought_30d         : 1 if CEO or CFO bought

    Log-scaling prevents large transactions dominating the feature.
    Zero-fills when no transaction data available.

    Note: For large-cap tech (FAANG), buy features will be near-zero
    due to pre-arranged 10b5-1 selling plans. Sell-side features and
    sell acceleration carry meaningful signal for these tickers.
    For mid/small-cap tickers, buy features carry strong signal.
    """
    zero_cols = [
        "insider_buy_count_30d", "insider_sell_count_30d",
        "insider_net_30d",       "insider_buy_value_30d",
        "insider_sell_value_30d", "insider_cluster_buy",
        "exec_bought_30d",
    ]

    if transactions.empty:
        result = pd.DataFrame(0, index=price_index, columns=zero_cols)
        result.index = pd.to_datetime(result.index).tz_localize(None)
        return result

    records = []
    for date in price_index:
        window_start = date - timedelta(days=INSIDER_WINDOW_DAYS)
        window = transactions[
            (transactions["trade_date"] >= window_start) &
            (transactions["trade_date"] <= date)
        ]
        buys  = window[window["is_buy"] == 1]
        sells = window[window["is_buy"] == 0]

        records.append({
            "date":                   date,
            "insider_buy_count_30d":  len(buys),
            "insider_sell_count_30d": len(sells),
            "insider_net_30d":        len(buys) - len(sells),
            "insider_buy_value_30d":  np.log1p(buys["value"].sum()),
            "insider_sell_value_30d": np.log1p(sells["value"].sum()),
            "insider_cluster_buy":    int(buys["insider"].nunique() >= 2)
                                      if not buys.empty else 0,
            "exec_bought_30d":        int(buys["is_exec"].sum() > 0)
                                      if not buys.empty else 0,
        })

    result = pd.DataFrame(records).set_index("date")
    result.index = pd.to_datetime(result.index).tz_localize(None)
    return result


def fetch_insider_transactions(ticker: str) -> pd.DataFrame:
    """
    Insider transaction pipeline for a ticker.

    Two-level caching:
        Level 1 — daily features CSV exists → load and return
        Level 2 — raw transactions CSV exists → recompute features only

    EDGAR only called when no cache exists at all.
    """
    raw_cache  = data_path(f"{ticker}_insider_raw.csv")
    daily_path = data_path(f"{ticker}_insider_daily.csv")

    # Level 1 — features already computed
    if os.path.exists(daily_path):
        daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        print(f"  Loaded insider features from cache ({len(daily)} rows)")
        return daily

    # Level 2 — raw transactions cached, recompute features
    if os.path.exists(raw_cache):
        print("  Loading raw transactions from cache...")
        transactions = pd.read_csv(raw_cache, parse_dates=["trade_date"])
        transactions["trade_date"] = pd.to_datetime(
            transactions["trade_date"]
        ).dt.tz_localize(None)
    else:
        # Fetch from EDGAR
        transactions = fetch_insider_raw(ticker)
        if not transactions.empty:
            transactions.to_csv(raw_cache, index=False)
            print(f"  Cached {len(transactions)} transactions")

    # Load price index for feature alignment
    raw_path = data_path(f"{ticker}_raw.csv")
    if not os.path.exists(raw_path):
        print(f"  Warning: {ticker}_raw.csv not found — skipping insider")
        return pd.DataFrame()

    price_df    = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    price_index = pd.to_datetime(price_df.index).tz_localize(None)

    daily = compute_insider_features(transactions, price_index)
    daily.to_csv(daily_path)
    print(f"  Saved insider features ({len(daily)} rows)")
    return daily


# ── UTILITIES ─────────────────────────────────────────────────────────

def get_company_name(ticker: str) -> str:
    """Fetch full company name from yfinance. Falls back to ticker."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def save_data(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to data/ directory as CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(data_path(filename))
    print(f"  Saved {filename} ({len(df)} rows)")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────

def run_data_pipeline(tickers: list = None) -> None:
    """
    Full data pipeline. All stages cache to disk — reruns are fast.

    Stage 1: OHLCV          — always refetches (fast, ~2s per ticker)
    Stage 2: Macro          — always refetches (fast, shared)
    Stage 3: Earnings       — always refetches (fast)
    Stage 4: Trends         — skips if CSV exists (429 rate limit protection)
    Stage 5: GDELT/FinBERT  — two-level cache, FinBERT skipped if complete
    Stage 6: Insider        — skips if daily CSV exists
    """
    if tickers is None:
        tickers = PRETRAINED_TICKERS

    print(f"\nStarting pipeline for: {tickers}")
    print(f"Training window: {TRAIN_START} → {get_train_end('default')}\n")

    # ── Stage 1: OHLCV ───────────────────────────────────────────────
    print("=" * 50)
    print("Stage 1: OHLCV")
    print("=" * 50)
    for ticker in tickers:
        try:
            save_data(fetch_market_data(ticker), f"{ticker}_raw.csv")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    # ── Stage 2: Macro ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Stage 2: Macro (VIX, TNX)")
    print("=" * 50)
    try:
        save_data(fetch_macro_data(), "macro.csv")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Stage 3: Earnings ────────────────────────────────────────────
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

    # ── Stage 4: Google Trends (cached) ──────────────────────────────
    print("\n" + "=" * 50)
    print("Stage 4: Google Trends")
    print("=" * 50)
    for ticker in tickers:
        try:
            if os.path.exists(data_path(f"{ticker}_trends.csv")):
                print(f"  {ticker}: cache hit")
                continue
            company_name = get_company_name(ticker)
            trends       = fetch_trends_data(ticker, company_name)
            if not trends.empty:
                save_data(trends, f"{ticker}_trends.csv")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    # ── Stage 5: GDELT + FinBERT (two-level cached) ───────────────────
    print("\n" + "=" * 50)
    print("Stage 5: GDELT Sentiment (FinBERT-scored)")
    print("=" * 50)
    for ticker in tickers:
        try:
            company_name = get_company_name(ticker)
            print(f"Processing {ticker} ({company_name})...")
            fetch_gdelt_sentiment(ticker, company_name)
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    # ── Stage 6: Insider Transactions (cached) ────────────────────────
    print("\n" + "=" * 50)
    print("Stage 6: Insider Transactions (SEC EDGAR)")
    print("=" * 50)
    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")
            fetch_insider_transactions(ticker)
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Pipeline complete:")
    print("=" * 50)
    for f in sorted(os.listdir(DATA_DIR)):
        size = os.path.getsize(data_path(f)) / 1024
        print(f"  {f:<45} {size:>8.1f} KB")


if __name__ == "__main__":
    run_data_pipeline()