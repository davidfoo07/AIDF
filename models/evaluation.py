import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def evaluate_ranking(test_df):
    """
    Evaluates ranking quality by comparing predicted scores vs actual forward Sharpe.
    """
    results = []

    for date, group in test_df.groupby("date"):
        actual = group["forward_sharpe"].values
        predicted = group["predicted_score"].values

        # Spearman rank correlation
        if len(set(actual)) > 1 and len(set(predicted)) > 1:
            corr, _ = spearmanr(actual, predicted)
        else:
            corr = 0.0

        # Top-1 accuracy: did we pick the best portfolio?
        actual_best = group.loc[group["forward_sharpe"].idxmax(), "portfolio_id"]
        predicted_best = group.loc[group["predicted_score"].idxmax(), "portfolio_id"]
        top1_correct = 1 if actual_best == predicted_best else 0

        results.append({
            "date": date,
            "spearman_corr": corr,
            "top1_correct": top1_correct,
        })

    eval_df = pd.DataFrame(results)


    return eval_df