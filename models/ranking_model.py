import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "momentum_20", "momentum_60",
    "volatility_20", "volatility_60",
    "avg_volume_20",
]


def prepare_data(portfolio_features: pd.DataFrame, forward_sharpe: pd.DataFrame):
    """
    Merges features with target, splits into train/test by time.
    
    Returns:
        X_train, y_train, group_train, X_test, y_test, group_test, test_df
    """
    # Merge features and target on 'date' and 'portfolio_id'
    merged = pd.merge(portfolio_features, forward_sharpe, on=["date", "portfolio_id"], how="inner")
    
    # Drop NaN rows
    merged = merged.dropna()
    
    # Sort by date then portfolio_id
    merged = merged.sort_values(by=["date", "portfolio_id"])
    
    # Split by time — e.g., train on dates before 2024-01-01, test on after
    train_dates = merged["date"] < "2024-01-01"
    test_dates = merged["date"] >= "2024-01-01"
    
    # Extract X (the 5 feature columns) and y (forward_sharpe) for train and test
    X_train = merged.loc[train_dates, FEATURE_COLS].copy()
    y_train = merged.loc[train_dates, "forward_sharpe"]
    X_test = merged.loc[test_dates, FEATURE_COLS].copy()
    y_test = merged.loc[test_dates, "forward_sharpe"]

    # Scale the features
    scaler = StandardScaler()
    X_train[FEATURE_COLS] = scaler.fit_transform(X_train[FEATURE_COLS])
    X_test[FEATURE_COLS] = scaler.transform(X_test[FEATURE_COLS])
   
    # Build group arrays — count portfolios per date
    group_train = merged.loc[train_dates].groupby("date").size().values
    group_test = merged.loc[test_dates].groupby("date").size().values

    return X_train, y_train, group_train, X_test, y_test, group_test, merged.loc[test_dates]


def train_ranking_model(X_train, y_train, group_train):
    """
    Trains XGBoost with ranking objective.
    
    Returns:
        trained model
    """
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Set group
    dtrain.set_group(group_train)
    
    # Set params with objective = "rank:pairwise"
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "learning_rate": 0.05,    
        "max_depth": 3,           
        "min_child_weight": 5,    
        "subsample": 0.8,         
        "colsample_bytree": 0.8,  
        "seed": 42,
    }

    # Train model
    model = xgb.train(params, dtrain, num_boost_round=100)

    return model


def predict_scores(model, X_test):
    """
    Predicts ranking scores for test data.
    
    Returns:
        array of scores
    """
    dtest = xgb.DMatrix(X_test)
    scores = model.predict(dtest)
    return scores