"""Projection layer that maps pooled features into the semantic memory space.

This module provides a multi-layer perceptron with non-linear activation and dropout
regularization to project high-dimensional pooled backbone representations into the
lower-dimensional semantic memory space.
"""

import torch.nn as nn


class MemoryProjection(nn.Module):
    """Feed-forward projection network bridging encoder features to memory dimensions."""

    def __init__(
        self,
        input_dim=768,
        memory_dim=256,
        dropout=0.1,
    ):
        """Initialize the projection network.

        Args:
            input_dim: Dimensionality of the incoming pooled encoder representations.
            memory_dim: Target dimensionality of the semantic memory space.
            dropout: Dropout probability applied after the intermediate activation.
        """
        super().__init__()

        # Multi-layer perceptron with intermediate dimension reduction and regularization
        self.layers = nn.Sequential(

            # First linear projection to an intermediate hidden dimension
            nn.Linear(
                input_dim,
                512,
            ),

            # Non-linear activation function
            nn.GELU(),

            # Dropout layer for regularization during training
            nn.Dropout(dropout),

            # Final linear projection down to the memory dimension
            nn.Linear(
                512,
                memory_dim,
            ),
        )

    def forward(
        self,
        x,
    ):
        """Project input representations into the semantic memory embedding space.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Projected tensor of shape (batch_size, memory_dim).
        """
        # Pass input representations through the sequential projection layers
        return self.layers(x)
