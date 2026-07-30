"""Learned semantic memory module with multi-head retrieval and gating."""

import math

import torch
import torch.nn as nn


class SemanticMemory(nn.Module):

    def __init__(
        self,
        memory_dim=256,
        num_heads=4,
        slots_per_head=16,
    ):
        super().__init__()

        assert (
            memory_dim % num_heads == 0
        ), "memory_dim must be divisible by num_heads"

        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.slots_per_head = slots_per_head

        self.head_dim = memory_dim // num_heads

        self.query = nn.Linear(
            memory_dim,
            memory_dim,
            bias=False,
        )

        self.keys = nn.Parameter(
            torch.empty(
                num_heads,
                slots_per_head,
                self.head_dim,
            )
        )

        self.values = nn.Parameter(
            torch.empty(
                num_heads,
                slots_per_head,
                self.head_dim,
            )
        )

        self.output = nn.Linear(
            memory_dim,
            memory_dim,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                memory_dim * 2,
                memory_dim,
            ),
            nn.Sigmoid(),
        )

        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

        nn.init.xavier_uniform_(self.query.weight)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(
        self,
        x,
    ):

        batch_size = x.size(0)

        q = self.query(x)

        q = q.view(
            batch_size,
            self.num_heads,
            self.head_dim,
        )

        outputs = []

        scale = math.sqrt(self.head_dim)

        for h in range(self.num_heads):

            scores = (
                torch.matmul(
                    q[:, h],
                    self.keys[h].T,
                )
                / scale
            )

            weights = torch.softmax(
                scores,
                dim=-1,
            )

            retrieved = torch.matmul(
                weights,
                self.values[h],
            )

            outputs.append(
                retrieved,
            )

        retrieved = torch.cat(
            outputs,
            dim=-1,
        )

        retrieved = self.output(
            retrieved,
        )
        gate = self.gate(
            torch.cat(
                [
                    x,
                    retrieved,
                ],
                dim=-1,
            )
        )

        memory = (
            gate * retrieved
            + (1.0 - gate) * x
        )

        return memory