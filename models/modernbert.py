"""ModernBERT backbone wrapper used as the feature extractor.

This module encapsulates a pretrained ModernBERT transformer encoder, providing
flexible parameter freezing for transfer learning and extracting contextual token
representations from input text sequences.
"""

from transformers import AutoModel
import torch.nn as nn


class Backbone(nn.Module):
    """Pretrained transformer encoder backbone for sequence feature extraction."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        unfreeze_last_n_layers: int = 0,
    ):
        """Initialize the ModernBERT backbone.

        Args:
            model_name: HuggingFace model identifier for the pretrained backbone.
            unfreeze_last_n_layers: Number of final transformer layers to keep trainable.
                Set to 0 to freeze the entire encoder.
        """
        super().__init__()

        # Load the pretrained transformer encoder model
        self.encoder = AutoModel.from_pretrained(
            model_name,
        )

        # Freeze all backbone parameters by default to preserve pretrained representations
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Optionally unfreeze the top N transformer layers for fine-tuning
        if unfreeze_last_n_layers > 0:

            # Enable gradient updates for the specified top layers
            for layer in self.encoder.layers[-unfreeze_last_n_layers:]:

                for param in layer.parameters():
                    param.requires_grad = True

            # Also unfreeze the final LayerNorm
            for param in self.encoder.final_norm.parameters():
                param.requires_grad = True

        # Store the hidden embedding dimension for downstream layers
        self.hidden_size = self.encoder.config.hidden_size

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        """Encode tokenized input sequences into contextual token representations.

        Args:
            input_ids: Tensor of input token IDs of shape (batch_size, sequence_length).
            attention_mask: Binary attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor of last hidden states of shape (batch_size, sequence_length, hidden_size).
        """
        # Pass inputs through the transformer encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Return token-level contextual representations
        return outputs.last_hidden_state
