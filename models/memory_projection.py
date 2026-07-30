"""Projection layer that maps pooled features into the semantic memory space."""

import torch.nn as nn


class MemoryProjection(nn.Module):

    def __init__(
        self,
        input_dim=768,
        memory_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                input_dim,
                512,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                memory_dim,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.layers(x)