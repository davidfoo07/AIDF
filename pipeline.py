import pandas as pd
from core.universe import PortfolioUniverse
from data.data_loader import download_market_data
from features.stock_features import compute_stock_features
from features.portfolio_features import compute_portfolio_features
from targets.sharpe_target import compute_forward_sharpe
from models.ranking_model import prepare_data, train_ranking_model, predict_scores
from conviction.conviction_scores import (
    softmax_conviction,
    minmax_conviction,
    zscore_conviction,
    rank_conviction,
    signal_to_noise_conviction,
)


def main():
    # Step 1: Load portfolios
    print("=" * 60)
    print("STEP 1: Loading portfolios...")
    print("=" * 60)
    universe = PortfolioUniverse.make_many("sample_portfolios.json")
    universe.resolve_portfolios_inplace()
    print(f"Number of portfolios: {len(universe.portfolios)}")
    print(f"Tickers: {universe.all_tickers()}")
    print(f"Example weights (portfolio 1): {universe.portfolios[0].resolved_portfolio_weights}")

    # Step 2: Download market data
    print("\n" + "=" * 60)
    print("STEP 2: Downloading market data...")
    print("=" * 60)
    prices, volumes = download_market_data(universe)
    print(f"Prices shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Prices head:\n{prices.head()}")

    # Step 3: Compute stock features
    print("\n" + "=" * 60)
    print("STEP 3: Computing stock features...")
    print("=" * 60)
    stock_features = compute_stock_features(prices, volumes)
    for name, df in stock_features.items():
        print(f"  {name}: shape={df.shape}")
    print(f"\nMomentum 20 sample:\n{stock_features['momentum_20'].dropna().head()}")

    # Step 4: Aggregate to portfolio features
    print("\n" + "=" * 60)
    print("STEP 4: Aggregating to portfolio features...")
    print("=" * 60)
    portfolio_features = compute_portfolio_features(stock_features, universe.portfolios)
    print(f"Portfolio features shape: {portfolio_features.shape}")
    print(f"Columns: {list(portfolio_features.columns)}")
    print(f"Head:\n{portfolio_features.head(10)}")

    # Step 5: Compute forward Sharpe (target)
    print("\n" + "=" * 60)
    print("STEP 5: Computing forward Sharpe ratio...")
    print("=" * 60)
    forward_sharpe = compute_forward_sharpe(prices, universe.portfolios)
    print(f"Forward Sharpe shape: {forward_sharpe.shape}")
    print(f"Head:\n{forward_sharpe.dropna().head(10)}")

    # Step 6: Prepare data and train model
    print("\n" + "=" * 60)
    print("STEP 6: Preparing data and training XGBoost ranking model...")
    print("=" * 60)
    X_train, y_train, group_train, X_test, y_test, group_test, test_df = prepare_data(portfolio_features, forward_sharpe)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Group sizes (first 5): {group_train[:5]}")
    print(f"Number of training dates: {len(group_train)}")
    print(f"Number of test dates: {len(group_test)}")

    print("\nTraining model...")
    model = train_ranking_model(X_train, y_train, group_train)
    print("Model training complete!")

    # Step 7: Predict scores
    print("\n" + "=" * 60)
    print("STEP 7: Predicting scores...")
    print("=" * 60)
    scores = predict_scores(model, X_test)
    test_df = test_df.copy()
    test_df["predicted_score"] = scores
    print(f"Scores shape: {scores.shape}")
    print(f"Score range: {scores.min():.4f} to {scores.max():.4f}")
    print(f"First 10 scores: {scores[:10]}")

    # Step 8: Apply conviction methods per date
    print("\n" + "=" * 60)
    print("STEP 8: Computing conviction scores...")
    print("=" * 60)
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
    print(f"Conviction DataFrame shape: {conviction_df.shape}")
    print(f"Columns: {list(conviction_df.columns)}")

    # Step 9: Print sample results
    print("\n" + "=" * 60)
    print("STEP 9: Sample Results")
    print("=" * 60)
    sample_date = conviction_df["date"].iloc[0]
    sample = conviction_df[conviction_df["date"] == sample_date]

    print(f"\nDate: {sample_date}")
    print("-" * 50)
    for _, row in sample.iterrows():
        print(f"\n  {row['portfolio_id']}:")
        print(f"    Predicted Score:{row['predicted_score']:.4f}")
        print(f"    Softmax Conviction:{row['softmax']:.4f}")
        print(f"    MinMax Conviction:    {row['minmax']:.4f}")
        print(f"    Z-Score Conviction:   {row['zscore']:.4f}")
        print(f"    Rank Conviction:      {row['rank']:.4f}")
        print(f"    Signal/Noise:         {row['signal_to_noise']:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()