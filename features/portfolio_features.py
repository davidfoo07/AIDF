import pandas as pd
from typing import Dict, List
from core.portfolios import Portfolio


def compute_portfolio_features(
    stock_features: Dict[str, pd.DataFrame],
    portfolios: List[Portfolio],
) -> pd.DataFrame:
    """
    Aggregates stock-level features into portfolio-level features.
    
    Returns:
        DataFrame with columns: 
            ['date', 'portfolio_id', 'momentum_20', 'momentum_60', 
             'volatility_20', 'volatility_60', 'avg_volume_20']"""

    portfolio_features = []

    for portfolio in portfolios:
        weights = portfolio.resolved_portfolio_weights
        row = pd.DataFrame()
        row["date"] = stock_features["momentum_20"].index
        row["portfolio_id"] = portfolio.portfolio_id

        for feature_name, df in stock_features.items():
            weighted = sum(df[ticker] * weight for ticker, weight in weights.items())
            row[feature_name] = weighted

        portfolio_features.append(row)

    return pd.concat(portfolio_features).reset_index(drop=True)