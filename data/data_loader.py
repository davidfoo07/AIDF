import yfinance as yf
import pandas as pd
from typing import Tuple
from core.universe import PortfolioUniverse


def download_market_data(
    universe: PortfolioUniverse,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads daily OHLCV data for all tickers in the universe.
    
    Returns:
        prices:DataFrame with columns=tickers, index=dates (Adj Close)
        volumes: DataFrame with columns=tickers, index=dates
    """
    tickers = universe.all_tickers()
    data = yf.download(tickers, start=start, end=end)
    
    prices = data["Close"].dropna()
    volumes = data["Volume"].dropna()

    # print(f"prices: {prices.shape[0]} rows, {prices.shape[1]} columns \n")
    # print(prices.head())
    # print(f"volumes: {volumes.shape[0]} rows, {volumes.shape[1]} columns\n")
    # print(volumes.head())

    return prices, volumes

