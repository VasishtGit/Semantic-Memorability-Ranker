"""Regression metrics used to evaluate model predictions."""

import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr


def regression_metrics(
    prediction,
    target,
):
    """Compute MAE, RMSE, Pearson, and Spearman metrics for predictions."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)

    mae = np.mean(
        np.abs(
            prediction - target
        )
    )

    rmse = np.sqrt(
        np.mean(
            (prediction - target) ** 2
        )
    )

    pearson = pearsonr(
        prediction,
        target,
    )[0]

    spearman = spearmanr(
        prediction,
        target,
    )[0]

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }