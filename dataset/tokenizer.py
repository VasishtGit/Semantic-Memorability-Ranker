"""Tokenizer wrapper for paragraph/clause memorability inputs."""

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
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

    def __call__(
        self,
        paragraph: str,
        target_clause: str,
    ):
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

        clause_mask = torch.tensor(
            [
                1 if sequence_id == 1 else 0
                for sequence_id in sequence_ids
            ],
            dtype=torch.long,
        )

        encoded["clause_mask"] = clause_mask.unsqueeze(0)

        return {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }