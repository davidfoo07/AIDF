import numpy as np
import pandas as pd


def softmax_conviction(scores):
    # weights sum to 1, use when want to allocate an amount of fund to each portfolio based on conviction
    # subtract max for numerical stability (prevents overflow when scores are large)
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def minmax_conviction(scores):
    # scale to 0-1, use to preserve relative differences but want to have a normalized range (eg: for blending with other signals)
    score_range = np.max(scores) - np.min(scores)
    if score_range == 0:
        return np.ones_like(scores) / len(scores)  # equal conviction if all scores are the same
    return (scores - np.min(scores)) / score_range


def zscore_conviction(scores):
    # standardize scores (can have -ve and +ve values), use when want to decide to go LONG or SHORT based on sign of score
    std = np.std(scores)
    if std == 0:
        return np.zeros_like(scores)  # zero conviction if no spread
    return (scores - np.mean(scores)) / std


def rank_conviction(scores):
    # rank scores, use when want to order portfolios without assuming linearity (eg: top 20% or bottom 20%)
    return pd.Series(scores).rank(pct=True).values


def signal_to_noise_conviction(scores, volatilities):
    # divide signal by volatility, use when want to prefer more stable portfolios
    safe_vol = np.where(volatilities == 0, 1e-8, volatilities)  # avoid division by zero
    return scores / safe_vol