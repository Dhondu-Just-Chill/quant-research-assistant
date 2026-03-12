# src/ml_forecasting.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os


# ── FEATURE ENGINEERING ──────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer technical indicators from raw OHLCV data.
    These become the features our ML model learns from.
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

    # ── Volatility: ATR (Average True Range)
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

    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target label:
    1 = price goes UP tomorrow
    0 = price goes DOWN tomorrow
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


# ── MODEL TRAINING ────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Select feature columns and split into X, y."""
    feature_cols = [
        "price_to_sma20", "price_to_sma50", "sma20_to_sma50",
        "macd", "macd_signal", "macd_histogram",
        "rsi",
        "bb_width", "bb_position",
        "atr",
        "volume_ratio",
        "return_1d", "return_5d", "return_10d"
    ]

    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols


def walk_forward_split(X, y, test_size=0.2):
    """
    Split data chronologically — NEVER shuffle financial time series.
    First 80% = training, last 20% = testing.
    """
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, ticker: str):
    """Evaluate model and plot feature importance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n {ticker} Model Evaluation:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    # Feature importance plot
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=True)

    feat_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{ticker} — Feature Importance")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = f"outputs/{ticker}_feature_importance.png"
    plt.savefig(path, dpi=150)
    print(f" Feature importance chart saved to {path}")
    plt.show()

    return accuracy


def save_model(model, ticker: str):
    """Save trained model to disk."""
    os.makedirs("models", exist_ok=True)
    path = f"models/{ticker}_xgb_model.pkl"
    joblib.dump(model, path)
    print(f" Model saved to {path}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────

def run_ml_pipeline(ticker: str) -> dict:
    """Full ML pipeline for a given ticker."""

    # Load data
    path = f"data/{ticker}_raw.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Feature engineering + labels
    df = add_technical_indicators(df)
    df = create_labels(df)

    # Prepare features
    X, y, feature_cols = prepare_features(df)

    # Split chronologically
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)

    print(f"\n {ticker} Training Info:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    print(f"  Features:         {len(feature_cols)}")

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test, ticker)

    # Save model
    save_model(model, ticker)

    return {
        "ticker": ticker,
        "accuracy": round(accuracy * 100, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_cols
    }


if __name__ == "__main__":
    for ticker in ["AAPL", "SPY"]:
        results = run_ml_pipeline(ticker)
        print(f"\n {ticker} pipeline complete — Accuracy: {results['accuracy']}%")
        print("---")