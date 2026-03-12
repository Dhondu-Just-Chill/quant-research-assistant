# src/statistical_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns and simple returns."""
    df = df.copy()
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def compute_core_stats(df: pd.DataFrame, ticker: str) -> dict:
    """
    Compute core quant statistics on the asset.
    
    Covers: risk, return, distribution shape, normality test
    """
    r = df["log_return"]

    # Annualized metrics (252 trading days)
    ann_return = r.mean() * 252
    ann_volatility = r.std() * np.sqrt(252)
    sharpe_ratio = ann_return / ann_volatility  # assumes risk-free rate = 0

    # Drawdown
    cumulative = (1 + df["simple_return"]).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Value at Risk (95% confidence, 1-day)
    var_95 = np.percentile(r, 5)

    # Distribution shape
    skewness = stats.skew(r)
    kurt = stats.kurtosis(r)  # excess kurtosis (normal = 0)

    # Normality test (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(r)
    is_normal = jb_pvalue > 0.05

    results = {
        "ticker": ticker,
        "annualized_return": round(ann_return * 100, 2),   # in %
        "annualized_volatility": round(ann_volatility * 100, 2),  # in %
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown * 100, 2),  # in %
        "var_95_1day": round(var_95 * 100, 2),  # in %
        "skewness": round(skewness, 4),
        "excess_kurtosis": round(kurt, 4),
        "jarque_bera_pvalue": round(jb_pvalue, 4),
        "returns_are_normal": is_normal
    }

    return results


def plot_analysis(df: pd.DataFrame, ticker: str) -> None:
    """Generate and save a 4-panel analysis chart."""
    os.makedirs("outputs", exist_ok=True)
    r = df["log_return"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{ticker} — Statistical Analysis", fontsize=16)

    # 1. Price history
    axes[0, 0].plot(df.index, df["Close"], color="steelblue")
    axes[0, 0].set_title("Price History")
    axes[0, 0].set_ylabel("Price (USD)")

    # 2. Daily log returns
    axes[0, 1].plot(df.index, r, color="orange", alpha=0.7)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Daily Log Returns")

    # 3. Return distribution vs normal
    axes[1, 0].hist(r, bins=60, density=True, alpha=0.6, color="steelblue", label="Actual")
    xmin, xmax = axes[1, 0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, r.mean(), r.std()), 
                    "r--", linewidth=2, label="Normal dist")
    axes[1, 0].set_title("Return Distribution")
    axes[1, 0].legend()

    # 4. Cumulative returns
    cumulative = (1 + df["simple_return"]).cumprod()
    axes[1, 1].plot(df.index, cumulative, color="green")
    axes[1, 1].set_title("Cumulative Returns")
    axes[1, 1].set_ylabel("Growth of $1")

    plt.tight_layout()
    path = f"outputs/{ticker}_analysis.png"
    plt.savefig(path, dpi=150)
    print(f"✅ Chart saved to {path}")
    plt.show()


def run_analysis(ticker: str) -> dict:
    """Load saved data, run full analysis, return stats dict."""
    path = f"data/{ticker}_raw.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = compute_returns(df)
    stats_dict = compute_core_stats(df, ticker)
    plot_analysis(df, ticker)
    return stats_dict


if __name__ == "__main__":
    for ticker in ["AAPL", "SPY"]:
        results = run_analysis(ticker)
        print(f"\n📊 {ticker} Stats:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("---")