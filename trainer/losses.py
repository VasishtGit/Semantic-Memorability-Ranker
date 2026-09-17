"""Loss functions for memorability ranking."""

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
        super().__init__()

        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight

        self.regression_loss = nn.SmoothL1Loss(
            beta=0.1
        )

    def forward(
        self,
        predictions,
        targets,
        story_ids,
    ):
        regression_loss = self.regression_loss(
            predictions,
            targets,
        )

        ranking_losses = []

        unique_stories = torch.unique(
            story_ids
        )

        for story_id in unique_stories:
            mask = story_ids == story_id

            story_predictions = predictions[
                mask
            ]

            story_targets = targets[
                mask
            ]

            if len(story_predictions) < 2:
                continue

            prediction_diff = (
                story_predictions.unsqueeze(1)
                - story_predictions.unsqueeze(0)
            )

            target_diff = (
                story_targets.unsqueeze(1)
                - story_targets.unsqueeze(0)
            )

            # Only compare pairs with different target ordering.
            pair_mask = target_diff != 0

            if not pair_mask.any():
                continue

            target_sign = torch.sign(
                target_diff[pair_mask]
            )

            prediction_diff = prediction_diff[
                pair_mask
            ]

            pair_loss = F.softplus(
                -target_sign * prediction_diff
            )

            # Weight pairs according to target-score difference.
            pair_weights = torch.abs(
                target_diff[pair_mask]
            )

            pair_weights = (
                pair_weights
                / pair_weights.mean().clamp_min(1e-8)
            )

            pair_loss = (
                pair_loss
                * pair_weights
            )

            ranking_losses.append(
                pair_loss.mean()
            )

        if ranking_losses:
            ranking_loss = torch.stack(
                ranking_losses
            ).mean()
        else:
            ranking_loss = predictions.new_tensor(
                0.0
            )

        return (
            self.regression_weight
            * regression_loss
            + self.ranking_weight
            * ranking_loss
        )


def get_loss(name="ranking"):
    """Return the requested memorability loss."""

    name = name.lower()

    if name == "ranking":
        return MemorabilityRankingLoss()

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    if name == "huber":
        return nn.SmoothL1Loss(
            beta=0.1
        )

    raise ValueError(
        f"Unknown loss: {name}"
    )