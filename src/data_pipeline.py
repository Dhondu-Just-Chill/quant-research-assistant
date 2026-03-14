# src/data_pipeline.py

import yfinance as yf
import pandas as pd
import os


def fetch_market_data(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker.

    Args:
        ticker   : stock symbol e.g. 'AAPL', 'SPY'
        period   : lookback period e.g. '3y', '1y', '6mo'
        interval : candle size e.g. '1d', '1wk'

    Returns:
        Clean DataFrame with OHLCV columns indexed by date.
    """
    print(f"Fetching OHLCV data for {ticker}...")
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Flatten multi-level columns if present (yfinance quirk)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    print(f"  Retrieved {len(df)} rows for {ticker}")
    return df


def fetch_macro_data(period: str = "3y") -> pd.DataFrame:
    """
    Fetch macro context features shared across all tickers.

    Fetches:
        VIX (^VIX) : market fear / volatility regime indicator
        TNX (^TNX) : 10-year treasury yield / interest rate environment

    Both are fetched for the same period as stock data so they
    can be merged by date index in the feature engineering step.

    Additional derived features:
        vix_change : day-over-day change in VIX (direction of fear)
        tnx_change : day-over-day change in yield (direction of rates)
        vix_ma20   : 20-day average VIX (smoothed regime indicator)
    """
    print("Fetching macro data (VIX, TNX)...")

    vix = yf.download("^VIX", period=period, interval="1d", auto_adjust=True)
    tnx = yf.download("^TNX", period=period, interval="1d", auto_adjust=True)

    # Flatten multi-level columns if present
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.get_level_values(0)

    macro = pd.DataFrame(index=vix.index)
    macro["vix"]        = vix["Close"]
    macro["tnx"]        = tnx["Close"]

    # Rate of change — direction matters more than absolute level
    macro["vix_change"] = macro["vix"].pct_change()
    macro["tnx_change"] = macro["tnx"].pct_change()

    # Smoothed VIX — reduces daily noise, better regime signal
    macro["vix_ma20"]   = macro["vix"].rolling(20).mean()

    macro.dropna(inplace=True)
    macro.index = pd.to_datetime(macro.index)

    print(f"  Retrieved {len(macro)} rows of macro data")
    return macro


def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Fetch historical earnings announcement dates for a ticker.

    Uses yfinance's earnings_dates property which returns upcoming
    and recent past earnings dates.

    Returns a DataFrame with a single 'earnings_date' column
    containing confirmed announcement dates, saved to disk for
    the feature engineering step to compute proximity features.

    Note: yfinance earnings data coverage varies by ticker and may
    not have complete historical data for all periods.
    """
    print(f"Fetching earnings dates for {ticker}...")
    try:
        t = yf.Ticker(ticker)

        # earnings_dates returns a DataFrame indexed by date
        earnings = t.earnings_dates
        if earnings is None or earnings.empty:
            print(f"  No earnings data available for {ticker}")
            return pd.DataFrame(columns=["earnings_date"])

        # Extract just the dates, drop future dates with no reported EPS
        dates = earnings.dropna(subset=["Reported EPS"]).index
        dates = pd.to_datetime(dates).tz_localize(None)  # remove timezone

        df = pd.DataFrame({"earnings_date": sorted(dates)})
        print(f"  Retrieved {len(df)} earnings dates for {ticker}")
        return df

    except Exception as e:
        print(f"  Could not fetch earnings for {ticker}: {e}")
        return pd.DataFrame(columns=["earnings_date"])


def save_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame to the data/ directory as CSV.

    Args:
        df       : DataFrame to save
        filename : filename without path e.g. 'AAPL_raw.csv'
    """
    os.makedirs("data", exist_ok=True)
    path = f"data/{filename}"
    df.to_csv(path)
    print(f"  Saved to {path}")


if __name__ == "__main__":
    tickers = ["AAPL", "SPY"]

    # Fetch and save OHLCV data for each ticker
    for ticker in tickers:
        df = fetch_market_data(ticker)
        save_data(df, f"{ticker}_raw.csv")

    # Fetch and save macro data once — shared across all tickers
    macro = fetch_macro_data()
    save_data(macro, "macro.csv")

    # Fetch and save earnings dates per ticker
    for ticker in tickers:
        earnings = fetch_earnings_dates(ticker)
        if not earnings.empty:
            save_data(earnings, f"{ticker}_earnings.csv")

    print("\nData pipeline complete.")
    print("Files saved:")
    for f in os.listdir("data"):
        print(f"  data/{f}")