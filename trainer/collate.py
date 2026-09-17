"""Collator that converts batches of samples into model-ready tensors."""

import torch

from dataset.tokenizer import MemorabilityTokenizer


class MemorabilityCollator:
    """Batch samples into tensors for the memorability regression model."""

    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
    ):
        self.tokenizer = MemorabilityTokenizer(
            model_name=model_name,
            max_length=max_length,
        )

    def __call__(
        self,
        batch,
    ):
        paragraphs = [
            sample["paragraph"]
            for sample in batch
        ]

        target_clauses = [
            sample["target_clause"]
            for sample in batch
        ]

        labels = torch.tensor(
            [
                sample["label"]
                for sample in batch
            ],
            dtype=torch.float32,
        )

        story_ids = torch.tensor(
            [
                sample["narrative_id"]
                for sample in batch
            ],
            dtype=torch.long,
        )

        encoded = self.tokenizer.tokenizer(
            paragraphs,
            target_clauses,
            truncation=True,
            padding=True,
            max_length=self.tokenizer.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Create a mask identifying target-clause tokens.
        clause_masks = []

        for batch_idx in range(
            len(batch)
        ):
            sequence_ids = encoded.sequence_ids(
                batch_idx
            )

            clause_mask = torch.tensor(
                [
                    1 if sequence_id == 1 else 0
                    for sequence_id in sequence_ids
                ],
                dtype=torch.long,
            )

            clause_masks.append(
                clause_mask
            )

        encoded["clause_mask"] = torch.stack(
            clause_masks,
            dim=0,
        )

        encoded["labels"] = labels

        encoded["story_ids"] = story_ids

        return encoded