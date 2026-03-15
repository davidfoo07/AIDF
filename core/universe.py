import json
from typing import List

from .payloads import PortfolioPayload
from .portfolios import Portfolio


class PortfolioUniverse:
    def __init__(self, portfolios: List[Portfolio]):
        self.portfolios = portfolios

    @classmethod
    def make_many(cls, filepath: str) -> "PortfolioUniverse":
        with open(filepath, "r") as f:
            raw = json.load(f)

        payloads = PortfolioPayload.make_many(raw)
        portfolios = [Portfolio.make(payload) for payload in payloads]

        return cls(portfolios)

    def resolve_portfolios_inplace(self) -> None:
        for portfolio in self.portfolios:
            portfolio.resolve_portfolio_inplace()

    def all_tickers(self) -> List[str]:
        tickers = set()
        for portfolio in self.portfolios:
            tickers.update(portfolio.tickers())
        return sorted(tickers)
