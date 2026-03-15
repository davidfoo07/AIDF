import pandas as pd
from core.universe import PortfolioUniverse
from data.data_loader import download_market_data
from features.stock_features import compute_stock_features
from features.portfolio_features import compute_portfolio_features
from targets.sharpe_target import compute_forward_sharpe
from models.ranking_model import prepare_data, train_ranking_model, predict_scores
from models.evaluation import evaluate_ranking
from conviction.conviction_scores import (
    softmax_conviction,
    minmax_conviction,
    zscore_conviction,
    rank_conviction,
    signal_to_noise_conviction,
)


def main():
    # load portfolios from JSON
    universe = PortfolioUniverse.make_many("sample_portfolios.json")
    universe.resolve_portfolios_inplace()
    print(f"Loaded {len(universe.portfolios)} portfolios with {len(universe.all_tickers())} unique tickers")

    # download price and volume data
    prices, volumes = download_market_data(universe)
    print(f"Downloaded data: {prices.shape[0]} days, {prices.shape[1]} tickers")

    # compute stock-level features then aggregate to portfolio level
    stock_features = compute_stock_features(prices, volumes)
    portfolio_features = compute_portfolio_features(stock_features, universe.portfolios)
    print(f"Portfolio features: {portfolio_features.shape}")

    # compute forward 20-day sharpe ratio as ranking target
    forward_sharpe = compute_forward_sharpe(prices, universe.portfolios)

    # split data, train xgboost ranking model
    X_train, y_train, group_train, X_test, y_test, group_test, test_df = prepare_data(portfolio_features, forward_sharpe)
    print(f"Train: {X_train.shape[0]} rows ({len(group_train)} dates), Test: {X_test.shape[0]} rows ({len(group_test)} dates)")

    model = train_ranking_model(X_train, y_train, group_train)

    # predict ranking scores on test set
    scores = predict_scores(model, X_test)
    test_df = test_df.copy()
    test_df["predicted_score"] = scores
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # evaluate ranking quality
    eval_df = evaluate_ranking(test_df)

    # apply conviction score methods per date
    all_convictions = []
    for date, group in test_df.groupby("date"):
        group_scores = group["predicted_score"].values

        row = group[["date", "portfolio_id", "predicted_score"]].copy()
        row["softmax"] = softmax_conviction(group_scores)
        row["minmax"] = minmax_conviction(group_scores)
        row["zscore"] = zscore_conviction(group_scores)
        row["rank"] = rank_conviction(group_scores)
        row["signal_to_noise"] = signal_to_noise_conviction(
            group_scores, group["volatility_20"].values
        )
        all_convictions.append(row)

    conviction_df = pd.concat(all_convictions).reset_index(drop=True)

    # show results for first test date
    sample_date = conviction_df["date"].iloc[0]
    sample = conviction_df[conviction_df["date"] == sample_date]

    print(f"\nConviction scores for {sample_date.date()}:")
    print(sample[["portfolio_id", "predicted_score", "softmax", "minmax", "zscore", "rank", "signal_to_noise"]].to_string(index=False))


if __name__ == "__main__":
    main()