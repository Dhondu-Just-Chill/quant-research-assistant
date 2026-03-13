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


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer technical features from raw OHLCV data.

    Transforms raw price/volume data into meaningful signals the ML model
    can learn from. Raw prices are scale-dependent and non-stationary —
    these derived features normalize that and capture market structure.

    Feature groups:
        - Moving averages / price ratios : trend direction
        - MACD                           : trend momentum and crossovers
        - RSI                            : overbought / oversold momentum
        - Bollinger Bands                : volatility and price extremes
        - ATR                            : average daily volatility range
        - Volume ratio                   : unusual trading activity
        - Lag returns                    : recent price momentum
        - OHLC features                  : intraday structure and pressure
    """
    df = df.copy()

    # --- Moving Averages ---
    # SMA: equal weight to all days in window
    # EMA: more weight on recent days, reacts faster to price changes
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # Price position relative to moving averages
    # Ratios are scale-free — works the same for a $5 or $500 stock
    # > 1.0 means price is above the average (bullish), < 1.0 means below (bearish)
    df["price_to_sma20"] = df["Close"] / df["sma_20"]
    df["price_to_sma50"] = df["Close"] / df["sma_50"]
    df["sma20_to_sma50"] = df["sma_20"] / df["sma_50"]  # short vs long term trend

    # --- MACD (Moving Average Convergence Divergence) ---
    # macd line     : difference between short and long term momentum
    # signal line   : smoothed macd, used to detect crossovers
    # histogram     : gap between macd and signal — positive = bullish momentum building
    df["macd"]           = df["ema_12"] - df["ema_26"]
    df["macd_signal"]    = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- RSI (Relative Strength Index) ---
    # Measures speed and magnitude of recent price changes
    # Range: 0-100. Above 70 = overbought (may fall), below 30 = oversold (may rise)
    delta    = df["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()   # average of up days
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean() # average of down days
    df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

    # --- Bollinger Bands ---
    # Bands expand during high volatility, contract during low volatility
    # bb_width    : how wide the bands are — measure of current volatility regime
    # bb_position : where price sits within the bands (0=lower, 0.5=middle, 1=upper)
    bb_std            = df["Close"].rolling(20).std()
    df["bb_upper"]    = df["sma_20"] + 2 * bb_std
    df["bb_lower"]    = df["sma_20"] - 2 * bb_std
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # --- ATR (Average True Range) ---
    # True range captures the full daily price movement including overnight gaps
    # ATR = 14-day average of true range — proxy for daily volatility
    tr = pd.concat([
        df["High"] - df["Low"],                          # intraday range
        (df["High"] - df["Close"].shift(1)).abs(),        # gap up scenario
        (df["Low"]  - df["Close"].shift(1)).abs()         # gap down scenario
    ], axis=1).max(axis=1)                                # worst case of the three
    df["atr"] = tr.rolling(14).mean()

    # --- Volume Ratio ---
    # Raw volume is meaningless across stocks — ratios normalize it
    # > 1.0 means today's volume is above average (significant move)
    # < 1.0 means below average (low conviction move)
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Lag Returns ---
    # Recent price momentum over different lookback windows
    # Captures whether the stock has been trending recently
    df["return_1d"]  = df["Close"].pct_change(1)   # yesterday's move
    df["return_5d"]  = df["Close"].pct_change(5)   # last week's move
    df["return_10d"] = df["Close"].pct_change(10)  # last two weeks' move

    # --- OHLC Derived Features ---
    # These extract intraday structure that Close alone cannot capture

    # overnight_gap : sentiment carried from previous close to today's open
    # positive = gapped up (bullish overnight), negative = gapped down (bearish)
    df["overnight_gap"]  = df["Open"] / df["Close"].shift(1) - 1

    # intraday_range : how volatile was the day relative to price level
    # high value = wide swings, low value = tight range day
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]

    # close_position : where did price close within the day's range
    # 1.0 = closed at the high (strong buying pressure)
    # 0.0 = closed at the low  (strong selling pressure)
    # 0.5 = closed in the middle (indecision)
    df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])

    # upper_shadow : how much buyers were rejected above the body
    # large upper shadow = sellers pushed price back down from intraday highs
    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]

    # lower_shadow : how much sellers were rejected below the body
    # large lower shadow = buyers pushed price back up from intraday lows
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    # Remove rows with NaN values created by rolling windows and shifts
    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target for next-day price direction.

    Label = 1 if tomorrow's close > today's close (price goes up)
    Label = 0 if tomorrow's close <= today's close (price goes down)

    shift(-1) looks one row forward — this is what makes it a prediction target.
    The last row is dropped since it has no next-day close to compare against.

    Design choice: predicting direction (classification) rather than exact price
    (regression) because direction is sufficient for trading decisions and is a
    cleaner, more learnable signal on small datasets.
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select the full default feature matrix from the engineered dataframe.

    Returns X (feature matrix), y (target labels), and the feature column list.
    The feature list is returned explicitly so downstream functions know which
    columns the model was trained on — critical when saving/loading models.

    Note: automatic feature selection in select_features() may produce a
    ticker-specific subset of these columns after pruning low-signal features.
    """
    feature_cols = [
        "price_to_sma20", "price_to_sma50", "sma20_to_sma50",
        "macd", "macd_signal", "macd_histogram",
        "rsi", "bb_width", "bb_position", "atr",
        "volume_ratio", "return_1d", "return_5d", "return_10d",
        "overnight_gap", "intraday_range", "close_position",
        "upper_shadow", "lower_shadow"
    ]
    return df[feature_cols], df["target"], feature_cols


def walk_forward_split(X, y, test_size=0.2):
    """
    Split data chronologically into training and test sets.

    NEVER shuffle time series data — shuffling causes data leakage by allowing
    the model to train on future data and test on past data, producing inflated
    accuracy that completely collapses in production.

    Correct approach: train on the first 80% of time, test on the last 20%.
    The model only ever sees past data during training, tested on genuinely
    unseen future data — the only honest evaluation for time series.
    """
    split = int(len(X) * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def tune_hyperparameters(X_train, y_train) -> dict:
    """
    Find optimal XGBoost hyperparameters using GridSearchCV with TimeSeriesSplit.

    GridSearchCV exhaustively tries every combination in param_grid and returns
    the combination that produced the best cross-validation accuracy.

    Critical: uses TimeSeriesSplit instead of standard KFold. Standard KFold
    shuffles data across folds which causes data leakage in time series.
    TimeSeriesSplit always trains on past folds and validates on future folds:

        Fold 1: [Train──────][Val]
        Fold 2: [Train────────────][Val]
        Fold 3: [Train──────────────────][Val]

    n_jobs=-1 uses all available CPU cores to parallelize the search.

    Hyperparameters being tuned:
        n_estimators     : number of trees to build
        max_depth        : how deep each tree can grow (controls overfitting)
        learning_rate    : how much each tree contributes to final prediction
        subsample        : fraction of rows each tree sees (regularization)
        colsample_bytree : fraction of features each tree sees (regularization)
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
    Train an XGBoost classifier with the given hyperparameters.

    XGBoost builds an ensemble of decision trees using gradient boosting:
        - Each tree learns from the residual errors of the previous trees
        - Trees are shallow (controlled by max_depth) to avoid overfitting
        - The final prediction is a weighted sum across all trees

    random_state=42 ensures reproducibility — same data always produces
    the same model, which is important for debugging and comparison.
    """
    model = XGBClassifier(**params, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    return model


def select_features(model, X_train, X_test, y_train, y_test, params: dict, ticker: str):
    """
    Automatically identify and remove low-signal features using permutation importance.

    Permutation importance works by:
        1. Taking a trained model and a held-out test set
        2. For each feature: shuffle its values randomly (breaks relationship with target)
        3. Measure how much accuracy drops after shuffling
        4. Importance = original_accuracy - shuffled_accuracy

    Interpretation:
        positive value : feature genuinely helps (accuracy drops when shuffled)
        zero           : feature is neutral
        negative value : feature actively hurts (shuffling it improves accuracy —
                         the model was learning spurious noise from it)

    Only features with negative importance are dropped — these are definitively
    harmful. Zero-importance features are kept since their scores have high
    variance on small datasets and they may be useful on different time windows.

    The pruned model is only adopted if it matches or beats the baseline accuracy —
    guaranteeing this step never makes things worse.

    Returns the best model, its feature list, and its accuracy.
    """
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))

    # Run permutation importance with 30 repeats for stable estimates
    perm     = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    perm_imp = pd.Series(perm.importances_mean, index=X_test.columns)

    # Identify features where shuffling actually improves accuracy
    to_drop   = perm_imp[perm_imp < 0].index.tolist()
    surviving = [f for f in X_test.columns if f not in to_drop]

    if not to_drop:
        print(f"  No features to drop - keeping all {len(X_test.columns)}")
        return model, list(X_test.columns), baseline_accuracy

    print(f"  Dropping {len(to_drop)} features: {to_drop}")

    # Retrain on surviving features with same hyperparameters
    pruned_model    = train_model(X_train[surviving], y_train, params)
    pruned_accuracy = accuracy_score(y_test, pruned_model.predict(X_test[surviving]))

    print(f"  Accuracy before pruning: {baseline_accuracy*100:.1f}%")
    print(f"  Accuracy after pruning:  {pruned_accuracy*100:.1f}%")

    if pruned_accuracy >= baseline_accuracy:
        print(f"  Pruning helped - using {len(surviving)} features")
        return pruned_model, surviving, pruned_accuracy
    else:
        print(f"  Pruning hurt accuracy - reverting to full feature set")
        return model, list(X_test.columns), baseline_accuracy


def evaluate_model(model, X_test, y_test, ticker: str) -> float:
    """
    Evaluate model performance and save a feature importance chart.

    Reports three evaluation metrics:
        accuracy  : overall fraction of correct predictions
        precision : when model predicts a class, how often is it right
        recall    : of all actual instances of a class, how many did it catch
        f1        : harmonic mean of precision and recall

    Accuracy alone is misleading when classes are imbalanced — precision
    and recall together reveal whether the model is biased toward one class.

    Feature importance plot shows each feature's contribution to reducing
    prediction error across all trees — useful for understanding what market
    signals the model relies on most.
    """
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {accuracy*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    # Plot feature importances sorted ascending for readability
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
    Persist the trained model and its feature list to disk.

    Both model and feature list are saved together in one file because the
    model can only make predictions on the exact features it was trained on.
    Saving them together prevents mismatches when loading in Week 3.

    Week 3 loading pattern:
        saved    = joblib.load(f"models/{ticker}_model.pkl")
        model    = saved["model"]
        features = saved["features"]
        prediction = model.predict(today_data[features])
    """
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, f"models/{ticker}_model.pkl")
    print(f"  Model saved to models/{ticker}_model.pkl")
    print(f"  Final feature count: {len(feature_cols)}")


def run_ml_pipeline(ticker: str) -> dict:
    """
    Execute the full ML pipeline for a given ticker symbol.

    Pipeline stages:
        1. Load raw OHLCV data from data/ directory
        2. Engineer 19 technical features from raw data
        3. Create binary next-day direction labels
        4. Split chronologically into train/test sets
        5. Tune hyperparameters via GridSearch + TimeSeriesSplit
        6. Train initial model on full feature set
        7. Run automatic feature selection via permutation importance
        8. Evaluate final model on held-out test set
        9. Save model + feature list to models/ directory

    The pipeline is fully ticker-agnostic — pass any valid ticker symbol
    and it will automatically tune, select features, and save a model.
    """
    # Load raw data saved by data_pipeline.py
    df = pd.read_csv(f"data/{ticker}_raw.csv", index_col=0, parse_dates=True)

    # Feature engineering then label creation
    # Order matters: labels use Close prices before any transformation
    df = create_labels(add_technical_indicators(df))

    # Build full feature matrix
    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)

    print(f"\n{ticker} | train: {len(X_train)} | test: {len(X_test)} | features: {len(feature_cols)}")

    # Stage 1: tune hyperparameters on training data only
    best_params = tune_hyperparameters(X_train, y_train)

    # Stage 2: train initial model with tuned params on full feature set
    model = train_model(X_train, y_train, best_params)

    # Stage 3: automatically prune low-signal features
    print(f"\n  Running automatic feature selection...")
    model, final_features, accuracy = select_features(
        model, X_train, X_test, y_train, y_test, best_params, ticker
    )

    # Stage 4: final evaluation on pruned feature set
    print(f"\n{ticker} Final Evaluation:")
    accuracy = evaluate_model(model, X_test[final_features], y_test, ticker)

    # Stage 5: save model + its specific feature list
    save_model(model, ticker, final_features)

    return {
        "ticker":         ticker,
        "accuracy":       round(accuracy * 100, 2),
        "best_params":    best_params,
        "final_features": final_features,
        "n_features":     len(final_features)
    }


if __name__ == "__main__":
    for ticker in ["AAPL", "SPY"]:
        results = run_ml_pipeline(ticker)
        print(f"\n{results['ticker']} complete")
        print(f"  Accuracy:   {results['accuracy']}%")
        print(f"  Features:   {results['n_features']}")
        print(f"  Params:     {results['best_params']}")
        print("---")