import pandas as pd
from typing import List
from core.portfolios import Portfolio


def compute_forward_sharpe(
    prices: pd.DataFrame,
    portfolios: List[Portfolio],
    forward_window: int = 20,
) -> pd.DataFrame:
    """
    Computes forward 20-day Sharpe ratio for each portfolio.
    
    Returns:
        DataFrame with columns: ['date', 'portfolio_id', 'forward_sharpe']
    """
    sharpe_ratios = []
    returns = prices.pct_change()

    for portfolio in portfolios:
        weights = portfolio.resolved_portfolio_weights
        portfolio_returns = sum(returns[ticker] * weight for ticker, weight in weights.items())

        rolling_mean = portfolio_returns.rolling(forward_window).mean().shift(-forward_window)
        rolling_std  = portfolio_returns.rolling(forward_window).std().shift(-forward_window)
        forward_sharpe = rolling_mean / rolling_std         

        row = pd.DataFrame({
            "date": prices.index,
            "portfolio_id": portfolio.portfolio_id,
            "forward_sharpe": forward_sharpe.values,
        })
        sharpe_ratios.append(row)  

    return pd.concat(sharpe_ratios).reset_index(drop=True)


def compute_backward_sharpe(
    prices: pd.DataFrame,
    portfolios: List[Portfolio],
    backward_window: int = 252,
) -> pd.DataFrame:
    """
    Computes backward 252-day Sharpe ratio for each portfolio.
    
    Returns:
        DataFrame with columns: ['date', 'portfolio_id', 'backward_sharpe'
    """
    sharpe_ratios = []
    returns = prices.pct_change()

    for portfolio in portfolios:
        weights = portfolio.resolved_portfolio_weights
        portfolio_returns = sum(returns[ticker] * weight for ticker, weight in weights.items())

        rolling_mean = portfolio_returns.rolling(backward_window).mean()
        rolling_std  = portfolio_returns.rolling(backward_window).std()
        backward_sharpe = rolling_mean / rolling_std         
        
        row = pd.DataFrame({
            "date": prices.index,
            "portfolio_id": portfolio.portfolio_id,
            "backward_sharpe": backward_sharpe.values,
        })
        sharpe_ratios.append(row)  

    return pd.concat(sharpe_ratios).reset_index(drop=True)