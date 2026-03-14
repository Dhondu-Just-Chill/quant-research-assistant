# src/data_pipeline.py

import yfinance as yf
import pandas as pd
import os

def fetch_market_data(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker.
    
    Args:
        ticker: Stock symbol e.g. 'AAPL', 'SPY'
        period: How far back to pull e.g. '1y', '2y', '6mo'
        interval: Candle size e.g. '1d', '1wk'
    
    Returns:
        Clean DataFrame with market data
    """
    print(f"Fetching data for {ticker}...")
    
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    
    if raw.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    
    # Clean up
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    
    print(f"✅ Retrieved {len(df)} rows for {ticker}")
    return df


def save_data(df: pd.DataFrame, ticker: str) -> None:
    """Save data to the data/ folder as CSV."""
    os.makedirs("data", exist_ok=True)
    path = f"data/{ticker}_raw.csv"
    df.to_csv(path)
    print(f"✅ Saved to {path}")


if __name__ == "__main__":
    # Test it with Apple and SPY (S&P 500 ETF)
    for ticker in ["AAPL", "SPY"]:
        df = fetch_market_data(ticker)
        save_data(df, ticker)
        print(df.tail())
        print("---")