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
            row[feature_name] = weighted.values

        portfolio_features.append(row)

    res = pd.concat(portfolio_features).reset_index(drop=True)
    res = res.dropna()

    # Only keep dates where ALL portfolios have data
    num_portfolios = len(portfolios)
    date_counts = res.groupby("date")["portfolio_id"].count()
    valid_dates = date_counts[date_counts == num_portfolios].index
    res = res[res["date"].isin(valid_dates)]

    return res.reset_index(drop=True)