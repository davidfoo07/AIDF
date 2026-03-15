from core.universe import PortfolioUniverse
from data.data_loader import download_market_data
from features.stock_features import compute_stock_features
from targets.sharpe_target import compute_forward_sharpe, compute_backward_sharpe

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

    # Test your data loader
    prices, volumes = download_market_data(universe)
    print("\nPrices shape:", prices.shape)
    print("Volumes shape:", volumes.shape)
    print("\nFirst 5 rows of prices:")
    print(prices.head())

    # Test your feature computation
    features = compute_stock_features(prices, volumes)
    print("\nComputed features:")
    for feature_name, df in features.items():
        print(f"{feature_name}: shape={df.shape}")
        print(df.dropna().head())  # Show first few rows of non-NA values

    # test your target computation
    forward_sharpe = compute_forward_sharpe(prices, universe.portfolios)
    print("\nForward Sharpe Ratios:")
    print(forward_sharpe)

    backward_sharpe = compute_backward_sharpe(prices, universe.portfolios)
    print("\nBackward Sharpe Ratios:")
    print(backward_sharpe)

if __name__ == "__main__":
    main()
