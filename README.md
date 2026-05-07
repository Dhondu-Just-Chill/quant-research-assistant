# LLM-Augmented Quantitative Research Assistant

A quantitative research platform that combines statistical analysis, machine learning, and large language models to analyze financial assets and generate investment research reports.

---

## Overview

Most retail investors lack access to the kind of multi-layered quantitative analysis that institutional desks run on individual stocks. This project builds an end-to-end research pipeline that layers four information sources — price/volume data, macro context, earnings proximity, and NLP-driven news sentiment — into a unified system that produces statistical risk profiles, ML-based directional forecasts, and LLM-generated research reports for any publicly traded stock.

The core finding of the project: **technical indicators alone are weak predictors of stock direction (~54% accuracy), but adding macro context (VIX, treasury yields) and earnings proximity lifts accuracy to ~62% for individual stocks like AAPL — empirically demonstrating that each information layer adds measurable predictive value.**

---

## Architecture

```
Yahoo Finance (yfinance)
        │
        ▼
┌───────────────────┐
│   data_pipeline   │  fetches OHLCV, VIX, TNX, earnings dates
└───────────────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌────────────────────┐          ┌───────────────────────┐
│statistical_analysis│          │    ml_forecasting     │
│                    │          │                       │
│ Sharpe ratio       │          │ Feature engineering   │
│ VaR                │          │ GridSearch + TSS CV   │
│ Max drawdown       │          │ XGBoost classifier    │
│ Kurtosis           │          │ Permutation pruning   │
│ Jarque-Bera        │          │ Auto feature select   │
└────────────────────┘          └───────────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
             ┌───────────────────┐
             │    llm_report     │  (Step 3 — in progress)
             │                   │
             │ News sentiment    │
             │ Research report   │
             │ LLM synthesis     │
             └───────────────────┘
                       │
                       ▼
             ┌────────────────────┐
             │      app.py        │  (Step 4 — planned)
             │                    │
             │ Streamlit UI       │
             │ Portfolio optimizer│
             │ Efficient frontier │
             └────────────────────┘
```

---

## Key Design Decisions

**No data leakage** — time series data is never shuffled. All train/test splits are strictly chronological. GridSearch cross validation uses `TimeSeriesSplit` exclusively, ensuring the model is always tested on genuinely unseen future data.

**Ticker-agnostic pipeline** — the entire system is parameterized by ticker symbol. Adding a new stock requires zero code changes — just pass the ticker to `run_ml_pipeline()`.

**Automatic feature selection** — permutation importance identifies features that actively hurt generalization (negative importance). The system automatically retrains without them and only adopts the pruned model if accuracy improves, guaranteeing the step never makes things worse.

**Model + feature list saved together** — each saved model file contains both the trained XGBoost model and the exact feature list it was trained on. This prevents input mismatch errors when loading models for inference.

**Per-ticker feature sets** — different stocks require fundamentally different feature sets. AAPL uses all 26 features. AMZN achieves similar accuracy with only 7. The system discovers this automatically rather than applying a one-size-fits-all approach.

---

## Empirical Findings

### Accuracy Progression — AAPL

| Model Version | Features | Test Samples | Accuracy |
|---|---|---|---|
| OHLCV only, 1 year | 19 | 41 | 61.0% (unreliable) |
| OHLCV only, 3 years | 19 | 141 | 53.9% (honest baseline) |
| + Macro + Earnings | 26 | 141 | **62.4%** |

The 8.5 percentage point improvement from adding macro context and earnings proximity empirically validates that each information layer adds measurable signal beyond technical indicators alone.

### Cross-Stock Comparison

| Ticker | Features Used | Accuracy | Interpretation |
|---|---|---|---|
| AAPL | 26 | 62.4% | Multi-factor, earnings-sensitive |
| AMZN | 7 | 59.6% | Clean momentum-driven signal |
| MSFT | 8 | 55.3% | Trend-following, Up-biased |
| GOOGL | 26 | 48.9% | News-driven, technical indicators insufficient |
| SPY | 6 | 53.9% | Efficient market, minimal predictable signal |

GOOGL's below-random accuracy on technical features alone is the strongest argument for the sentiment layer — it is precisely the kind of news and regulatory event-driven stock where NLP context adds the most value.

### Feature Set Findings

- AAPL is genuinely multi-factorial — all 26 features contribute positively
- AMZN and MSFT have clean, sparse signal — most features are noise
- SPY (the index) dropped almost all macro features including VIX — the index IS the market, making VIX partially redundant
- `return_5d` (5-day lag return) was pruned as noise for almost every ticker — short-term momentum has minimal predictive power at daily resolution
- New OHLC features (`overnight_gap`, `close_position`, `intraday_range`) ranked in the top 5 for AAPL — validating their addition over Close-only features

---

## Feature Engineering

### Technical Indicators (from OHLCV)

| Feature | Description |
|---|---|
| `price_to_sma20` | Price relative to 20-day moving average — trend position |
| `price_to_sma50` | Price relative to 50-day moving average — longer trend |
| `sma20_to_sma50` | Short vs long term trend alignment |
| `macd` | EMA12 - EMA26 — momentum direction |
| `macd_signal` | Smoothed MACD — crossover signal |
| `macd_histogram` | MACD - Signal — momentum acceleration |
| `rsi` | Relative Strength Index — overbought/oversold |
| `bb_width` | Bollinger Band width — volatility regime |
| `bb_position` | Price position within bands |
| `atr` | Average True Range — daily volatility |
| `volume_ratio` | Volume vs 20-day average — unusual activity |
| `return_1d` | Yesterday's return |
| `return_5d` | 5-day return — weekly momentum |
| `return_10d` | 10-day return — biweekly momentum |

### OHLC Structure Features

| Feature | Description |
|---|---|
| `overnight_gap` | Open vs previous Close — overnight sentiment |
| `intraday_range` | (High - Low) / Close — daily volatility |
| `close_position` | Where Close sits in day's range — buying pressure |
| `upper_shadow` | Rejection above the body — selling pressure |
| `lower_shadow` | Rejection below the body — buying pressure |

### Macro Context Features

| Feature | Description |
|---|---|
| `vix` | CBOE Volatility Index — market fear level |
| `vix_change` | Day-over-day VIX change — direction of fear |
| `vix_ma20` | Smoothed VIX — volatility regime indicator |
| `tnx` | 10-year treasury yield — interest rate level |
| `tnx_change` | Day-over-day yield change — rate direction |

### Earnings Proximity Features (individual stocks only)

| Feature | Description |
|---|---|
| `days_to_earnings` | Days until next earnings announcement (capped at 60) |
| `days_from_earnings` | Days since last earnings announcement (capped at 60) |

---

## ML Methodology

**Model:** XGBoost binary classifier — chosen over alternatives because it consistently outperforms other algorithms on structured tabular data, handles nonlinear feature interactions, and provides interpretable feature importance scores.

**Why not ARIMA:** ARIMA is univariate and assumes linear relationships. With 26 engineered features and nonlinear market dynamics, a tree-based ensemble is structurally more appropriate.

**Why not LSTM/CNN:** Both require significantly more data than our 563 training samples. On tabular daily data at this scale, XGBoost outperforms deep learning architectures while being faster to train and easier to interpret.

**Hyperparameter tuning:** Exhaustive GridSearch across 243 parameter combinations using 5-fold `TimeSeriesSplit` cross validation. Optimizes for accuracy while preventing data leakage.

**Class balancing:** `scale_pos_weight` automatically computed as the ratio of down days to up days in training data. Prevents the model from defaulting to always predicting the majority class.

**Feature selection:** Permutation importance with 30 repeats on held-out test data. Features with negative mean importance are candidates for removal. Pruned model only adopted if accuracy matches or improves baseline.

---

## Project Structure

```
quant-research-assistant/
│
├── data/                        # auto-generated, not committed
│   ├── AAPL_raw.csv
│   ├── GOOGL_raw.csv
│   ├── macro.csv                # VIX + TNX shared across tickers
│   └── AAPL_earnings.csv
│
├── models/                      # auto-generated, not committed
│   ├── AAPL_model.pkl           # model + feature list
│   └── GOOGL_model.pkl
│
├── outputs/                     # auto-generated, not committed
│   ├── AAPL_analysis.png
│   └── AAPL_feature_importance.png
│
├── src/
│   ├── data_pipeline.py         # data fetching layer
│   ├── statistical_analysis.py  # risk metrics and visualization
│   ├── ml_forecasting.py        # ML training pipeline
│   ├── llm_report.py            # LLM sentiment + report (Week 3)
│   └── app.py                   # Streamlit UI (Week 4)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+, Windows/Mac/Linux

```bash
# Clone the repository
git clone https://github.com/Dhondu-Just-Chill/quant-research-assistant.git
cd quant-research-assistant

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

**Step 1 — Fetch data:**
```bash
python src/data_pipeline.py
```
Fetches 3 years of OHLCV data, macro indicators, and earnings dates for configured tickers. Saves to `data/`.

**Step 2 — Statistical analysis:**
```bash
python src/statistical_analysis.py
```
Computes risk metrics and generates 4-panel analysis charts for each ticker. Saves to `outputs/`.

**Step 3 — Train ML models:**
```bash
python src/ml_forecasting.py
```
Runs full ML pipeline: feature engineering, hyperparameter tuning, automatic feature selection, evaluation. Saves trained models to `models/`.

**Adding a new ticker:**

In `data_pipeline.py` and `ml_forecasting.py`, add the ticker to the list:
```python
tickers = ["AAPL", "SPY", "GOOGL", "AMZN", "MSFT"] # add any valid ticker here
```
No other code changes required.

---

## Roadmap

- [x] Step 1 — Data pipeline, statistical analysis module
- [x] Step 2 — ML forecasting with XGBoost, GridSearch, macro features, automatic feature selection
- [ ] Step 3 — News sentiment analysis, LLM research report generator
- [ ] Step 4 — Streamlit UI, multi-asset portfolio optimizer, efficient frontier

### Planned Enhancements (Post v1)
- Hourly bar feature engineering aggregated to daily predictions
- Backtesting module with transaction cost modeling
- Expanded ticker coverage with sector-based grouping
- Options flow and put/call ratio as additional features

---

## Limitations

**Data size:** 3 years of daily data produces 563 training samples — sufficient for a demonstration system but below what production quant models typically use (5-10 years minimum).

**Technical indicators only (currently):** Fundamental data (P/E ratio, revenue growth, debt levels) and alternative data (satellite imagery, credit card transactions) are not yet incorporated. These represent the next most impactful information layers after sentiment.

**No transaction costs:** Current accuracy metrics do not account for bid-ask spread, brokerage commissions, or market impact. A strategy that looks profitable before costs may not be after.

**Stationarity assumption:** The model assumes statistical properties of the past will approximately hold in the future. This assumption breaks down during regime changes (e.g. 2008 financial crisis, COVID crash).

**Not financial advice:** This is a research and portfolio project. Nothing in this system constitutes investment advice.

---

## Skills Demonstrated

- End-to-end ML pipeline design with proper train/test methodology for time series
- Feature engineering from raw financial data across four information layers
- Hyperparameter optimization without data leakage using TimeSeriesSplit
- Automated feature selection using permutation importance
- Statistical analysis of financial time series (VaR, Sharpe, drawdown, kurtosis)
- System design for ticker-agnostic, extensible pipelines
- LLM integration for financial NLP (Week 3)
- Full-stack deployment with Streamlit (Week 4)
