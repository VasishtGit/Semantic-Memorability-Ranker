"""Attention-based pooling layer for selecting target-clause information.

This module computes attention weights specifically over tokens belonging to the
target clause, masking out surrounding context tokens, padding tokens, and special
tokens so that the pooled representation captures only clause-relevant semantics.
"""

import torch
import torch.nn as nn


class ClauseAttentionPooling(nn.Module):
    """Pools representations using only target-clause tokens."""

    def __init__(
        self,
        hidden_size: int = 768,
    ):
        """Initialize the clause attention pooling layer.

        Args:
            hidden_size: Feature dimension of the input token representations.
        """
        super().__init__()

        # Linear projection to compute scalar unnormalized attention scores per token
        self.score = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        hidden_states,
        clause_mask,
    ):
        """Aggregate token representations within the target clause using attention.

        Args:
            hidden_states: Contextual token representations of shape (batch_size, seq_len, hidden_size).
            clause_mask: Binary mask of shape (batch_size, seq_len) where 1 indicates
                a target-clause token and 0 indicates context, special, or padding tokens.

        Returns:
            Pooled clause representation tensor of shape (batch_size, hidden_size).
        """
        # Project hidden representations to scalar scores and remove trailing dimension
        scores = self.score(
            hidden_states,
        ).squeeze(-1)

        # Mask out non-clause tokens by assigning the minimum representable float value
        # so they receive approximately zero probability in softmax
        scores = scores.masked_fill(
            clause_mask == 0,
            torch.finfo(scores.dtype).min,
        )

        # Compute normalized attention weights over the sequence tokens
        weights = torch.softmax(
            scores,
            dim=1,
        )

        # Compute the attention-weighted sum of token representations across the sequence
        pooled = torch.sum(
            hidden_states * weights.unsqueeze(-1),
            dim=1,
        )

        # Return the aggregated clause representation
        return pooled
