"""Main neural network module that combines encoding, pooling, memory, and regression."""

import torch
import torch.nn as nn

from .modernbert import Backbone
from .clause_attention import ClauseAttentionPooling
from .memory_projection import MemoryProjection
from .semantic_memory import SemanticMemory


class NeuroDapt(nn.Module):

    def __init__(
        self,
        memory_dim=256,
        unfreeze_last_n_layers=0,
    ):
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
    ):

        # Produce contextual token embeddings from the input sequence.
        hidden = self.backbone(
            input_ids,
            attention_mask,
        )

        # Combine the token embeddings into a single clause-focused vector.
        pooled = self.pooling(
            hidden,
            attention_mask,
        )

        # Move the pooled representation into the memory embedding space.
        memory = self.memory_projection(
            pooled,
        )

        # Refine the representation using the learned semantic memory bank.
        memory = self.semantic_memory(
            memory,
        )

        # Predict a scalar memorability score and squash it to the $(0, 1)$ range.
        score = self.regression(
            memory,
        )

        score = torch.sigmoid(
            score,
        )

        return score.squeeze(-1)