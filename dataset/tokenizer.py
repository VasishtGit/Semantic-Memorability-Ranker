"""Tokenizer wrapper for paragraph/clause memorability inputs.

This module provides a specialized tokenizer wrapper that formats paragraph and target-clause
pairs into model inputs, automatically generating a binary clause mask using token sequence IDs
to distinguish clause tokens from surrounding narrative context and special tokens.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer


class MemorabilityTokenizer:
    """Encode paragraph and target-clause pairs for memorability prediction."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 512,
    ):
        """Initialize the tokenizer wrapper.

        Args:
            model_name: HuggingFace model identifier for loading the pretrained tokenizer.
            max_length: Maximum sequence length for padding and truncation.
        """
        self.max_length = max_length

        # Load the pretrained HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

    def __call__(
        self,
        paragraph: str,
        target_clause: str,
    ):
        """Tokenize a paragraph and clause pair and construct a clause selection mask.

        Args:
            paragraph: Full narrative context string.
            target_clause: Specific clause within the paragraph being evaluated.

        Returns:
            Dictionary containing input_ids, attention_mask, and clause_mask tensors.
        """
        # Tokenize sequence pair with fixed-length padding and truncation
        encoded = self.tokenizer(
            paragraph,
            target_clause,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # sequence_ids identifies which input each token came from:
        #   0    -> paragraph
        #   1    -> target clause
        #   None -> special tokens
        sequence_ids = encoded.sequence_ids(0)

        # Generate binary mask where 1 indicates target-clause tokens
        clause_mask = torch.tensor(
            [
                1 if sequence_id == 1 else 0
                for sequence_id in sequence_ids
            ],
            dtype=torch.long,
        )

        # Insert clause mask with matching batch dimension
        encoded["clause_mask"] = clause_mask.unsqueeze(0)

        # Remove the leading batch dimension for single-example output
        return {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
