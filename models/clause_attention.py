"""Attention-based pooling layer for selecting target-clause information."""

import torch
import torch.nn as nn


class ClauseAttentionPooling(nn.Module):
    """Pools representations using only target-clause tokens."""

    def __init__(
        self,
        hidden_size: int = 768,
    ):
        super().__init__()

        self.score = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        hidden_states,
        clause_mask,
    ):
        """
        hidden_states:
            (B, T, H)

        clause_mask:
            (B, T)
            1 = target clause
            0 = context/special/padding
        """

        scores = self.score(
            hidden_states,
        ).squeeze(-1)

        scores = scores.masked_fill(
            clause_mask == 0,
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            hidden_states * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled