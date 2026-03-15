import numpy as np
import pandas as pd
import xgboost as xgb


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
    feature_columns = [
        "momentum_20", "momentum_60",
        "volatility_20", "volatility_60",
        "avg_volume_20",
    ]
    X_train = merged.loc[train_dates, feature_columns]
    y_train = merged.loc[train_dates, "forward_sharpe"]
    X_test = merged.loc[test_dates, feature_columns]
    y_test = merged.loc[test_dates, "forward_sharpe"]
   
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
        "learning_rate": 0.1,
        "max_depth": 4,
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