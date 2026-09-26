"""Collator that converts batches of samples into model-ready tensors.

This module provides custom batch collation for PyTorch DataLoader instances, handling
dynamic sequence padding, label and metadata tensor packaging, and constructing aligned
clause selection masks across the batch.
"""

import torch

from dataset.tokenizer import MemorabilityTokenizer


class MemorabilityCollator:
    """Batch samples into tensors for the memorability regression model."""

    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
    ):
        """Initialize the batch collator with a tokenizer.

        Args:
            model_name: HuggingFace model identifier for the tokenizer.
            max_length: Maximum allowed sequence length for batched sequences.
        """
        # Internal tokenizer instance used for tokenizing pairs
        self.tokenizer = MemorabilityTokenizer(
            model_name=model_name,
            max_length=max_length,
        )

    def __call__(
        self,
        batch,
    ):
        """Transform a list of dataset sample dictionaries into batched model input tensors.

        Args:
            batch: List of sample dictionaries containing paragraph, target_clause,
                label, and narrative_id keys.

        Returns:
            Dictionary containing input_ids, attention_mask, clause_mask,
            labels, and story_ids tensors.
        """
        # Extract paragraph context strings across the batch
        paragraphs = [
            sample["paragraph"]
            for sample in batch
        ]

        # Extract target clause strings across the batch
        target_clauses = [
            sample["target_clause"]
            for sample in batch
        ]

        # Convert scalar memorability labels into a float tensor of shape (batch_size,)
        labels = torch.tensor(
            [
                sample["label"]
                for sample in batch
            ],
            dtype=torch.float32,
        )

        # Convert narrative IDs into a long tensor of shape (batch_size,)
        story_ids = torch.tensor(
            [
                sample["narrative_id"]
                for sample in batch
            ],
            dtype=torch.long,
        )

        # Tokenize paragraph-clause pairs with dynamic batch-level padding
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

        # Iterate over each sample in the batch to construct its clause mask
        for batch_idx in range(
            len(batch)
        ):
            # sequence_ids identifies which input segment each token belongs to:
            # 0 for paragraph, 1 for target clause, None for special tokens
            sequence_ids = encoded.sequence_ids(
                batch_idx
            )

            # Assign 1 to target clause tokens and 0 to all other tokens
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

        # Stack individual clause masks into a single batch tensor: (batch_size, seq_len)
        encoded["clause_mask"] = torch.stack(
            clause_masks,
            dim=0,
        )

        # Attach ground-truth labels and story identifiers to the batch dictionary
        encoded["labels"] = labels

        encoded["story_ids"] = story_ids

        return encoded
