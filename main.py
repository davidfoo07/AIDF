from core.universe import PortfolioUniverse


def main():
    universe = PortfolioUniverse.make_many("sample_portfolios.json")
    universe.resolve_portfolios_inplace()

    print("Number of portfolios:", len(universe.portfolios))
    print("Tickers:", universe.all_tickers())

    p = universe.portfolios[0]

    print("\nExample portfolio:")
    print("Name:", p.name)
    print("Weights:", p.resolved_portfolio_weights)
    print("Weight sum:", sum(p.resolved_portfolio_weights.values()))


if __name__ == "__main__":
    main()
