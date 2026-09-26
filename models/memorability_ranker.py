"""Main neural network module that combines encoding, pooling, memory, and regression.

This module defines the end-to-end Semantic Memorability Ranker architecture. It passes
tokenized paragraph-clause sequences through a ModernBERT backbone, applies clause-masked
attention pooling to isolate target-clause features, projects them into a latent memory space,
queries a multi-head learned semantic memory bank with residual gating, and outputs a
normalized memorability score via a sigmoid regression head.
"""

import torch
import torch.nn as nn

from .modernbert import Backbone
from .clause_attention import ClauseAttentionPooling
from .memory_projection import MemoryProjection
from .semantic_memory import SemanticMemory


class SemanticMemorabilityRanker(nn.Module):
    """End-to-end neural network for narrative clause memorability prediction and ranking."""

    def __init__(
        self,
        memory_dim=256,
        unfreeze_last_n_layers=0,
    ):
        """Initialize the memorability ranker pipeline.

        Args:
            memory_dim: Dimensionality of the intermediate projection and semantic memory.
            unfreeze_last_n_layers: Number of top ModernBERT transformer layers to unfreeze.
        """
        super().__init__()

        # Encode the input text with the pretrained ModernBERT backbone.
        self.backbone = Backbone(
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )

        # Aggregate token-level representations into a single clause-aware vector.
        self.pooling = ClauseAttentionPooling(
            hidden_size=self.backbone.hidden_size,
        )

        # Project the pooled embedding into the memory space.
        self.memory_projection = MemoryProjection(
            input_dim=self.backbone.hidden_size,
            memory_dim=memory_dim,
        )

        # Apply the semantic memory module to refine the representation.
        self.semantic_memory = SemanticMemory(
            memory_dim=memory_dim,
            num_heads=4,
            slots_per_head=16,
        )

        # Final regression head that predicts a memorability score.
        self.regression = nn.Linear(
            memory_dim,
            1,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        clause_mask,
    ):
        """Execute the forward pass to predict clause memorability scores.

        Args:
            input_ids: Tensor of input token IDs of shape (batch_size, sequence_length).
            attention_mask: Attention mask tensor of shape (batch_size, sequence_length).
            clause_mask: Binary mask of shape (batch_size, sequence_length) selecting target-clause tokens.

        Returns:
            Tensor of scalar memorability predictions in range [0, 1] of shape (batch_size,).
        """
        # Produce contextual token embeddings from ModernBERT.
        hidden = self.backbone(
            input_ids,
            attention_mask,
        )

        # Pool ONLY target-clause tokens.
        pooled = self.pooling(
            hidden,
            clause_mask,
        )

        # Project into memory space.
        memory = self.memory_projection(
            pooled,
        )

        # Retrieve from learned semantic memory.
        memory = self.semantic_memory(
            memory,
        )

        # Regression prediction.
        score = self.regression(
            memory,
        )

        # Constrain predictions to the [0, 1] interval
        score = torch.sigmoid(
            score,
        )

        # Squeeze trailing singleton dimension to yield a 1D batch score tensor
        return score.squeeze(-1)
