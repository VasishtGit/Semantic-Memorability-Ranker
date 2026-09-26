"""Regression metrics used to evaluate model predictions.

This module computes core statistical metrics for evaluating continuous memorability
predictions against ground-truth ratings, including absolute error, root mean squared error,
Pearson linear correlation, and Spearman monotonic rank correlation.
"""

import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr


def regression_metrics(
    prediction,
    target,
):
    """Compute MAE, RMSE, Pearson, and Spearman metrics for predictions.

    Args:
        prediction: Array-like sequence of predicted continuous values.
        target: Array-like sequence of ground-truth target values.

    Returns:
        Dictionary mapping metric names ('mae', 'rmse', 'pearson', 'spearman')
        to their computed floating-point values.
    """

    # Convert inputs to contiguous 1D NumPy arrays
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    # Compute Mean Absolute Error
    mae = np.mean(
        np.abs(
            prediction - target
        )
    )

    # Compute Root Mean Squared Error
    rmse = np.sqrt(
        np.mean(
            (prediction - target) ** 2
        )
    )

    # Compute Pearson linear correlation coefficient
    pearson = pearsonr(
        prediction,
        target,
    )[0]

    # Compute Spearman rank-order correlation coefficient
    spearman = spearmanr(
        prediction,
        target,
    )[0]

    # Return structured metric dictionary
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }
