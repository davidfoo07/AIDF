import pandas as pd
from typing import Dict


def compute_stock_features(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Computes stock-level features.
    
    Returns:
        dict with keys: 'momentum_20', 'momentum_60', 
                        'volatility_20', 'volatility_60', 
                        'avg_volume_20'
        Each value is a DataFrame with columns=tickers, index=dates.
    """
    features = {}

    # Momentum
    features["momentum_20"] = prices / prices.shift(20) - 1
    features["momentum_60"] = prices / prices.shift(60) - 1

    # Volatility
    daily_returns = prices.pct_change()
    features["volatility_20"] = daily_returns.rolling(window=20).std()
    features["volatility_60"] = daily_returns.rolling(window=60).std()

    # Average Volume
    features["avg_volume_20"] = volumes.rolling(window=20).mean()

    return features