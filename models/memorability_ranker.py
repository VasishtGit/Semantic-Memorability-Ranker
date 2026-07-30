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

        ##################################################
        # Backbone
        ##################################################

        self.backbone = Backbone(
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )

        ##################################################
        # Clause Attention Pooling
        ##################################################

        self.pooling = ClauseAttentionPooling(
            hidden_size=self.backbone.hidden_size,
        )

        ##################################################
        # Memory Projection
        ##################################################

        self.memory_projection = MemoryProjection(
            input_dim=self.backbone.hidden_size,
            memory_dim=memory_dim,
        )

        ##################################################
        # Semantic Memory
        ##################################################

        self.semantic_memory = SemanticMemory(
            memory_dim=memory_dim,
            num_heads=4,
            slots_per_head=16,
        )

        ##################################################
        # Regression Head
        ##################################################

        self.regression = nn.Linear(
            memory_dim,
            1,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        ##################################################
        # ModernBERT
        ##################################################

        hidden = self.backbone(
            input_ids,
            attention_mask,
        )

        ##################################################
        # Clause Attention Pooling
        ##################################################

        pooled = self.pooling(
            hidden,
            attention_mask,
        )

        ##################################################
        # Memory Projection
        ##################################################

        memory = self.memory_projection(
            pooled,
        )

        ##################################################
        # Semantic Memory
        ##################################################

        memory = self.semantic_memory(
            memory,
        )

        ##################################################
        # Regression
        ##################################################

        score = self.regression(
            memory,
        )

        score = torch.sigmoid(
            score,
        )

        return score.squeeze(-1)