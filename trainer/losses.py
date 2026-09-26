"""Loss functions for memorability ranking.

This module implements loss functions for training the memorability ranker, featuring a
composite objective that blends pointwise Smooth L1 (Huber) regression loss with a within-story
pairwise ranking loss to simultaneously optimize absolute memorability prediction and relative
clause ordering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemorabilityRankingLoss(nn.Module):
    """Combined regression and pairwise ranking loss."""

    def __init__(
        self,
        regression_weight=0.5,
        ranking_weight=0.5,
    ):
        """Initialize the composite ranking loss.

        Args:
            regression_weight: Scalar multiplier applied to the pointwise regression loss.
            ranking_weight: Scalar multiplier applied to the pairwise ranking loss.
        """
        super().__init__()

        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight

        # Smooth L1 (Huber) loss for robust pointwise regression
        self.regression_loss = nn.SmoothL1Loss(
            beta=0.1
        )

    def forward(
        self,
        predictions,
        targets,
        story_ids,
    ):
        """Compute the weighted sum of pointwise regression and pairwise ranking losses.

        Args:
            predictions: Predicted memorability tensor of shape (batch_size,).
            targets: Ground-truth memorability tensor of shape (batch_size,).
            story_ids: Story identifier tensor of shape (batch_size,).

        Returns:
            Scalar loss tensor representing the total combined loss.
        """
        # Calculate pointwise Smooth L1 regression loss across all batch items
        regression_loss = self.regression_loss(
            predictions,
            targets,
        )

        ranking_losses = []

        # Find unique story identifiers present in the current batch
        unique_stories = torch.unique(
            story_ids
        )

        # Compute pairwise ranking loss independently for each story group
        for story_id in unique_stories:
            mask = story_ids == story_id

            story_predictions = predictions[
                mask
            ]

            story_targets = targets[
                mask
            ]

            # At least two clauses from the same story are required to form pairs
            if len(story_predictions) < 2:
                continue

            # Compute all pairwise prediction differences: diff[i, j] = pred[i] - pred[j]
            prediction_diff = (
                story_predictions.unsqueeze(1)
                - story_predictions.unsqueeze(0)
            )

            # Compute all pairwise target differences: diff[i, j] = target[i] - target[j]
            target_diff = (
                story_targets.unsqueeze(1)
                - story_targets.unsqueeze(0)
            )

            # Only compare pairs with different target ordering.
            pair_mask = target_diff != 0

            # Skip if there are no valid distinguishable pairs
            if not pair_mask.any():
                continue

            # Direction of target preference (+1 if target[i] > target[j], -1 otherwise)
            target_sign = torch.sign(
                target_diff[pair_mask]
            )

            prediction_diff = prediction_diff[
                pair_mask
            ]

            # Softplus loss penalizing negative values of target_sign * prediction_diff
            pair_loss = F.softplus(
                -target_sign * prediction_diff
            )

            # Weight pairs according to target-score difference.
            pair_weights = torch.abs(
                target_diff[pair_mask]
            )

            # Normalize weights by their mean for stable gradient scaling
            pair_weights = (
                pair_weights
                / pair_weights.mean().clamp_min(1e-8)
            )

            # Apply magnitude weights to pair losses
            pair_loss = (
                pair_loss
                * pair_weights
            )

            # Record mean ranking loss for this story
            ranking_losses.append(
                pair_loss.mean()
            )

        # Average the per-story ranking losses across stories in the batch
        if ranking_losses:
            ranking_loss = torch.stack(
                ranking_losses
            ).mean()
        else:
            # Fall back to zero tensor on the appropriate device if no pairs exist
            ranking_loss = predictions.new_tensor(
                0.0
            )

        # Return weighted combination of regression and ranking objectives
        return (
            self.regression_weight
            * regression_loss
            + self.ranking_weight
            * ranking_loss
        )


def get_loss(name="ranking"):
    """Return the requested memorability loss.

    Args:
        name: Name of the loss function ('ranking', 'mse', 'mae', or 'huber').

    Returns:
        An instantiated PyTorch loss module.
    """

    name = name.lower()

    # Composite regression and pairwise ranking loss
    if name == "ranking":
        return MemorabilityRankingLoss()

    # Standard Mean Squared Error loss
    if name == "mse":
        return nn.MSELoss()

    # Mean Absolute Error (L1) loss
    if name == "mae":
        return nn.L1Loss()

    # Smooth L1 (Huber) regression loss
    if name == "huber":
        return nn.SmoothL1Loss(
            beta=0.1
        )

    raise ValueError(
        f"Unknown loss: {name}"
    )
