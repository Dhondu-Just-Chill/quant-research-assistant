# src/ml_forecasting.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib
import os


# ── FEATURE ENGINEERING ──────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer technical indicators from raw OHLCV data.
    Uses all four price points (OHLC) to maximize information extraction.
    """
    df = df.copy()

    # ── Trend: Simple & Exponential Moving Averages
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # Price position relative to moving averages
    df["price_to_sma20"] = df["Close"] / df["sma_20"]
    df["price_to_sma50"] = df["Close"] / df["sma_50"]
    df["sma20_to_sma50"] = df["sma_20"] / df["sma_50"]

    # ── Momentum: MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # ── Momentum: RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── Volatility: Bollinger Bands
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2 * bb_std
    df["bb_lower"] = df["sma_20"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── Volatility: ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # ── Volume
    df["volume_ma20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma20"]

    # ── Returns (lag features)
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # ── NEW: OHLC derived features ─────────────────────────────────

    # Overnight gap — sentiment carried from previous close to today's open
    df["overnight_gap"] = df["Open"] / df["Close"].shift(1) - 1

    # Intraday range — how volatile was the day relative to price
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]

    # Close position within day's range — buying/selling pressure
    # 1.0 = closed at high (strong buyers), 0.0 = closed at low (strong sellers)
    df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])

    # Upper shadow — how much buyers were rejected above close
    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]

    # Lower shadow — how much sellers were rejected below close
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target:
    1 = tomorrow's close > today's close (price goes up)
    0 = tomorrow's close <= today's close (price goes down)
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


# ── FEATURE PREPARATION ───────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Select all engineered feature columns and return X, y."""
    feature_cols = [
        # Trend
        "price_to_sma20", "price_to_sma50", "sma20_to_sma50",
        # Momentum
        "macd", "macd_signal", "macd_histogram",
        "rsi",
        # Volatility
        "bb_width", "bb_position",
        "atr",
        # Volume
        "volume_ratio",
        # Lag returns
        "return_1d", "return_5d", "return_10d",
        # NEW: OHLC features
        "overnight_gap", "intraday_range", "close_position",
        "upper_shadow", "lower_shadow"
    ]

    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols


def walk_forward_split(X, y, test_size=0.2):
    """Chronological split — never shuffle time series data."""
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test


# ── GRIDSEARCH WITH TIMESERIESSPLIT ───────────────────────────────────

def tune_hyperparameters(X_train, y_train) -> dict:
    """
    Find optimal hyperparameters using GridSearchCV with TimeSeriesSplit.
    TimeSeriesSplit ensures no future data leaks into training during CV.
    """
    print("  🔍 Running GridSearch (this may take ~30 seconds)...")

    # Parameter grid — all combinations will be tried
    param_grid = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [3, 4, 5, 6],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.7, 0.8, 0.9],
        "colsample_bytree":[0.7, 0.8, 0.9],
    }

    # TimeSeriesSplit — 5 folds, always training on past testing on future
    tscv = TimeSeriesSplit(n_splits=5)

    # Base model
    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    # GridSearch with TimeSeriesSplit as cross validator
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,       # use all CPU cores
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"  ✅ Best params found: {grid_search.best_params_}")
    print(f"  ✅ Best CV accuracy:  {grid_search.best_score_:.4f}")

    return grid_search.best_params_


# ── MODEL TRAINING ────────────────────────────────────────────────────

def train_model(X_train, y_train, params: dict = None):
    """Train XGBoost with either provided or default parameters."""
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    model = XGBClassifier(
        **params,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ── EVALUATION ────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, ticker: str):
    """Evaluate model performance and plot feature importance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    # Feature importance
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=True)

    feat_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{ticker} — Feature Importance (Tuned Model)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = f"outputs/{ticker}_feature_importance.png"
    plt.savefig(path, dpi=150)
    print(f"\n  ✅ Feature importance chart saved to {path}")
    plt.show()

    return accuracy


def save_model(model, ticker: str):
    """Persist trained model to disk for Week 3 LLM layer."""
    os.makedirs("models", exist_ok=True)
    path = f"models/{ticker}_xgb_model.pkl"
    joblib.dump(model, path)
    print(f"  ✅ Model saved to {path}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────

def run_ml_pipeline(ticker: str) -> dict:
    """Full ML pipeline: features → labels → tune → train → evaluate → save."""

    # Load raw data
    path = f"data/{ticker}_raw.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Feature engineering + label creation
    df = add_technical_indicators(df)
    df = create_labels(df)

    # Prepare feature matrix
    X, y, feature_cols = prepare_features(df)

    # Chronological split
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)

    print(f"\n🔧 {ticker} Pipeline:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    print(f"  Features:         {len(feature_cols)}")

    # Hyperparameter tuning
    best_params = tune_hyperparameters(X_train, y_train)

    # Train with best params
    model = train_model(X_train, y_train, params=best_params)

    # Evaluate
    print(f"\n📊 {ticker} Model Evaluation:")
    accuracy = evaluate_model(model, X_test, y_test, ticker)

    # Save
    save_model(model, ticker)

    return {
        "ticker": ticker,
        "accuracy": round(accuracy * 100, 2),
        "best_params": best_params,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_cols
    }


if __name__ == "__main__":
    for ticker in ["AAPL", "SPY"]:
        results = run_ml_pipeline(ticker)
        print(f"\n✅ {ticker} complete — Accuracy: {results['accuracy']}%")
        print(f"   Best params: {results['best_params']}")
        print("---")