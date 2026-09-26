"""Learned semantic memory module with multi-head retrieval and gating.

This module provides an explicit memory bank of learned key-value slots that can be
queried via multi-head attention. A learned sigmoid gate adaptively blends the retrieved
memory representations with the original input embeddings via a residual highway.
"""

import math

import torch
import torch.nn as nn


class SemanticMemory(nn.Module):
    """Multi-head key-value memory bank with adaptive residual gating."""

    def __init__(
        self,
        memory_dim=256,
        num_heads=4,
        slots_per_head=16,
    ):
        """Initialize the semantic memory module.

        Args:
            memory_dim: Total dimensionality of the memory representations.
            num_heads: Number of parallel memory attention heads.
            slots_per_head: Number of addressable memory slots per attention head.
        """
        super().__init__()

        # Ensure memory dimensionality can be evenly divided across heads
        assert (
            memory_dim % num_heads == 0
        ), "memory_dim must be divisible by num_heads"

        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.slots_per_head = slots_per_head

        # Dimensionality allocated to each attention head
        self.head_dim = memory_dim // num_heads

        # Linear projection mapping input vectors to query representations
        self.query = nn.Linear(
            memory_dim,
            memory_dim,
            bias=False,
        )

        # Learnable memory keys: (num_heads, slots_per_head, head_dim)
        self.keys = nn.Parameter(
            torch.empty(
                num_heads,
                slots_per_head,
                self.head_dim,
            )
        )

        # Learnable memory values: (num_heads, slots_per_head, head_dim)
        self.values = nn.Parameter(
            torch.empty(
                num_heads,
                slots_per_head,
                self.head_dim,
            )
        )

        # Output projection to combine multi-head retrieved representations
        self.output = nn.Linear(
            memory_dim,
            memory_dim,
        )

        # Gating network to compute an element-wise interpolation factor between input and memory
        self.gate = nn.Sequential(
            nn.Linear(
                memory_dim * 2,
                memory_dim,
            ),
            nn.Sigmoid(),
        )

        # Initialize memory keys and values with Xavier uniform initialization
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

        # Initialize query projection weights
        nn.init.xavier_uniform_(self.query.weight)

        # Initialize output projection weights and zero bias
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        # Initialize gate weights and zero bias
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(
        self,
        x,
    ):
        """Query memory slots and blend retrieved features with the input vector.

        Args:
            x: Input tensor of shape (batch_size, memory_dim).

        Returns:
            Gated memory output tensor of shape (batch_size, memory_dim).
        """
        batch_size = x.size(0)

        # Project input to query representations
        q = self.query(x)

        # Reshape queries into multi-head format: (batch_size, num_heads, head_dim)
        q = q.view(
            batch_size,
            self.num_heads,
            self.head_dim,
        )

        outputs = []

        # Scaling factor for dot-product attention
        scale = math.sqrt(self.head_dim)

        # Retrieve information independently from each attention head
        for h in range(self.num_heads):

            # Compute similarity scores between queries and memory slot keys
            scores = (
                torch.matmul(
                    q[:, h],
                    self.keys[h].T,
                )
                / scale
            )

            # Softmax normalize scores to obtain slot attention weights
            weights = torch.softmax(
                scores,
                dim=-1,
            )

            # Retrieve memory contents as a weighted sum of memory values
            retrieved = torch.matmul(
                weights,
                self.values[h],
            )

            outputs.append(
                retrieved,
            )

        # Concatenate retrieved vectors from all heads along the feature dimension
        retrieved = torch.cat(
            outputs,
            dim=-1,
        )

        # Project the concatenated multi-head representations
        retrieved = self.output(
            retrieved,
        )

        # Compute gating values by conditioning on both input and retrieved features
        gate = self.gate(
            torch.cat(
                [
                    x,
                    retrieved,
                ],
                dim=-1,
            )
        )

        # Convex combination of retrieved memory representation and original input
        memory = (
            gate * retrieved
            + (1.0 - gate) * x
        )

        return memory
