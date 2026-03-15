import pandas as pd

from core.universe import PortfolioUniverse

def main():
    universe = PortfolioUniverse.make_many("sample_portfolios.json")
    universe.resolve_portfolios_inplace()

    print("Number of portfolios:", len(universe.portfolios))
    print("Tickers:", universe.all_tickers())

if __name__ == "__main__":
    main()
