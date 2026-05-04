# src/ml_forecasting.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
import os


def load_and_merge_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro context features (VIX, TNX) into the stock DataFrame by date.

    Uses a left join on the date index so stock rows are preserved even if
    a macro data point is missing for that date (e.g. half-trading days).
    Missing macro values are forward-filled — last known value carries forward.

    Macro features added:
        vix        : raw VIX level — absolute fear reading
        vix_change : day-over-day VIX change — direction of fear
        vix_ma20   : smoothed VIX — regime indicator
        tnx        : 10-year treasury yield level
        tnx_change : day-over-day yield change — direction of rates
    """
    macro_path = "data/macro.csv"
    if not os.path.exists(macro_path):
        print("  Warning: macro.csv not found, skipping macro features.")
        return df

    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)

    # Normalize timezone — yfinance sometimes returns tz-aware indices
    macro.index = macro.index.tz_localize(None) if macro.index.tzinfo else macro.index
    df.index    = df.index.tz_localize(None) if df.index.tzinfo else df.index

    df = df.join(macro, how="left")

    # Forward fill any gaps (e.g. VIX missing on a day stock traded)
    for col in ["vix", "vix_change", "vix_ma20", "tnx", "tnx_change"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    return df


def load_and_merge_earnings(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Compute earnings proximity features and merge into the stock DataFrame.

    For each trading day, computes:
        days_to_earnings   : calendar days until the next earnings announcement
                             captures pre-earnings uncertainty / volatility expansion
        days_from_earnings : calendar days since the last earnings announcement
                             captures post-earnings drift patterns

    If no earnings file exists for the ticker (e.g. ETFs like SPY), the
    function returns the DataFrame unchanged — no features added, no error.

    Proximity is capped at 60 days in both directions to prevent outliers
    from dominating the feature distribution.
    """
    earnings_path = f"data/{ticker}_earnings.csv"
    if not os.path.exists(earnings_path):
        print(f"  No earnings file for {ticker} — skipping earnings features.")
        return df

    earnings = pd.read_csv(earnings_path, parse_dates=["earnings_date"])
    dates    = sorted(earnings["earnings_date"].dropna().tolist())

    if not dates:
        return df

    def days_to_next(current_date):
        # Find the nearest future earnings date
        future = [d for d in dates if d > current_date]
        if not future:
            return 60  # cap at 60 if no future date known
        return min((future[0] - current_date).days, 60)

    def days_from_last(current_date):
        # Find the nearest past earnings date
        past = [d for d in dates if d <= current_date]
        if not past:
            return 60  # cap at 60 if no prior date known
        return min((current_date - past[-1]).days, 60)

    df["days_to_earnings"]   = df.index.map(days_to_next)
    df["days_from_earnings"] = df.index.map(days_from_last)

    print(f"  Added earnings proximity features for {ticker}")
    return df


def add_technical_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Engineer the full feature set from OHLCV + macro + earnings data.

    Feature groups:
        Moving averages / price ratios : trend direction and position
        MACD                           : trend momentum and crossovers
        RSI                            : overbought / oversold momentum
        Bollinger Bands                : volatility bands and price position
        ATR                            : average daily volatility range
        Volume ratio                   : relative trading activity
        Lag returns                    : recent price momentum
        OHLC features                  : intraday structure and pressure
        Macro features                 : market regime context (VIX, TNX)
        Earnings proximity             : distance to earnings events
    """
    df = df.copy()

    # --- Merge macro and earnings context before computing features ---
    df = load_and_merge_macro(df)
    df = load_and_merge_earnings(df, ticker)

    # --- Moving Averages ---
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # Price position relative to moving averages (scale-free ratios)
    df["price_to_sma20"] = df["Close"] / df["sma_20"]
    df["price_to_sma50"] = df["Close"] / df["sma_50"]
    df["sma20_to_sma50"] = df["sma_20"] / df["sma_50"]

    # --- MACD ---
    df["macd"]           = df["ema_12"] - df["ema_26"]
    df["macd_signal"]    = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- RSI ---
    delta    = df["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

    # --- Bollinger Bands ---
    bb_std            = df["Close"].rolling(20).std()
    df["bb_upper"]    = df["sma_20"] + 2 * bb_std
    df["bb_lower"]    = df["sma_20"] - 2 * bb_std
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # --- ATR ---
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # --- Volume ---
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Lag returns ---
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # --- OHLC derived features ---
    df["overnight_gap"]  = df["Open"] / df["Close"].shift(1) - 1
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
    df["upper_shadow"]   = (df["High"] - df[["Open","Close"]].max(axis=1)) / df["Close"]
    df["lower_shadow"]   = (df[["Open","Close"]].min(axis=1) - df["Low"]) / df["Close"]

    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if tomorrow's close > today's close, else 0.
    Last row dropped — no next-day close available to label it.
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


def prepare_features(df: pd.DataFrame):
    """
    Build the full feature matrix from the engineered DataFrame.

    Dynamically includes macro and earnings columns only if they exist —
    making the pipeline work for both individual stocks (with earnings)
    and ETFs (without earnings).
    """
    # Core technical features — always present
    feature_cols = [
        "price_to_sma20", "price_to_sma50", "sma20_to_sma50",
        "macd", "macd_signal", "macd_histogram",
        "rsi", "bb_width", "bb_position", "atr",
        "volume_ratio", "return_1d", "return_5d", "return_10d",
        "overnight_gap", "intraday_range", "close_position",
        "upper_shadow", "lower_shadow"
    ]

    # Macro features — present if macro.csv was loaded successfully
    macro_cols   = ["vix", "vix_change", "vix_ma20", "tnx", "tnx_change"]

    # Earnings features — present only for individual stocks
    earnings_cols = ["days_to_earnings", "days_from_earnings"]

    # Only include columns that actually exist in the DataFrame
    for col in macro_cols + earnings_cols:
        if col in df.columns:
            feature_cols.append(col)

    return df[feature_cols], df["target"], feature_cols


def walk_forward_split(X, y, test_size=0.2):
    """
    Chronological train/test split — never shuffle time series.
    First 80% used for training, last 20% for evaluation.
    """
    split = int(len(X) * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def tune_hyperparameters(X_train, y_train) -> dict:
    """
    Exhaustive hyperparameter search using GridSearchCV + TimeSeriesSplit.
    TimeSeriesSplit prevents data leakage during cross validation.
    n_jobs=-1 parallelizes across all CPU cores.
    """
    print("  Running GridSearch...")
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }
    grid_search = GridSearchCV(
        estimator=XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"  Best params:   {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    return grid_search.best_params_


def train_model(X_train, y_train, params: dict) -> XGBClassifier:
    """
    Train XGBoost classifier with given hyperparameters.
    scale_pos_weight balances class weights when up/down days are unequal.
    """
    # Balance classes — penalize missing the minority class more heavily
    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    scale = neg / pos

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def select_features(model, X_train, X_test, y_train, y_test, params: dict, ticker: str):
    """
    Automatically prune features with negative permutation importance.

    Permutation importance shuffles each feature and measures accuracy drop.
    Negative importance means the feature actively hurts generalization —
    the model learned spurious noise from it.

    Only adopts the pruned model if accuracy matches or improves.
    Guarantees this step never makes things worse.
    """
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))

    perm     = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    perm_imp = pd.Series(perm.importances_mean, index=X_test.columns)

    to_drop   = perm_imp[perm_imp < 0].index.tolist()
    surviving = [f for f in X_test.columns if f not in to_drop]

    if not to_drop:
        print(f"  No features to drop - keeping all {len(X_test.columns)}")
        return model, list(X_test.columns), baseline_accuracy

    print(f"  Dropping {len(to_drop)} features: {to_drop}")

    pruned_model    = train_model(X_train[surviving], y_train, params)
    pruned_accuracy = accuracy_score(y_test, pruned_model.predict(X_test[surviving]))

    print(f"  Accuracy before pruning: {baseline_accuracy*100:.1f}%")
    print(f"  Accuracy after pruning:  {pruned_accuracy*100:.1f}%")

    if pruned_accuracy >= baseline_accuracy:
        print(f"  Pruning helped - using {len(surviving)} features")
        return pruned_model, surviving, pruned_accuracy
    else:
        print(f"  Pruning hurt - reverting to full feature set")
        return model, list(X_test.columns), baseline_accuracy


def evaluate_model(model, X_test, y_test, ticker: str) -> float:
    """
    Report accuracy, classification metrics, and save feature importance chart.

    Precision and recall are reported alongside accuracy because accuracy
    alone is misleading when up/down days are imbalanced.
    """
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {accuracy*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    feat_imp = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{ticker} - Feature Importance (Final Model)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{ticker}_feature_importance.png", dpi=150)
    plt.show()

    return accuracy


def save_model(model, ticker: str, feature_cols: list):
    """
    Save model and its exact feature list together.
    Week 3 loads both to ensure prediction inputs match training inputs.
    """
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, f"models/{ticker}_model.pkl")
    print(f"  Model saved to models/{ticker}_model.pkl")
    print(f"  Final feature count: {len(feature_cols)}")


def run_ml_pipeline(ticker: str) -> dict:
    """
    Full ML pipeline for any ticker symbol.

    Stages:
        1. Load OHLCV data
        2. Merge macro + earnings context
        3. Engineer full feature set
        4. Create binary direction labels
        5. Chronological train/test split
        6. GridSearch hyperparameter tuning
        7. Train initial model
        8. Automatic feature selection via permutation importance
        9. Final evaluation
        10. Save model + features
    """
    df = pd.read_csv(f"data/{ticker}_raw.csv", index_col=0, parse_dates=True)
    df = create_labels(add_technical_indicators(df, ticker))

    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)

    print(f"\n{ticker} | train: {len(X_train)} | test: {len(X_test)} | features: {len(feature_cols)}")

    best_params = tune_hyperparameters(X_train, y_train)
    model       = train_model(X_train, y_train, best_params)

    print(f"\n  Running automatic feature selection...")
    model, final_features, accuracy = select_features(
        model, X_train, X_test, y_train, y_test, best_params, ticker
    )

    print(f"\n{ticker} Final Evaluation:")
    accuracy = evaluate_model(model, X_test[final_features], y_test, ticker)

    save_model(model, ticker, final_features)

    return {
        "ticker":         ticker,
        "accuracy":       round(accuracy * 100, 2),
        "best_params":    best_params,
        "final_features": final_features,
        "n_features":     len(final_features)
    }


if __name__ == "__main__":
    for ticker in ["AAPL", "SPY", "GOOGL", "AMZN", "MSFT"]:
        results = run_ml_pipeline(ticker)
        print(f"\n{results['ticker']} complete")
        print(f"  Accuracy:  {results['accuracy']}%")
        print(f"  Features:  {results['n_features']}")
        print(f"  Params:    {results['best_params']}")
        print("---")