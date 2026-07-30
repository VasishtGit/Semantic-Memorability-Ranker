import torch

from dataset.tokenizer import NeuroDaptTokenizer


class NeuroDaptCollator:

    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
    ):

        self.tokenizer = NeuroDaptTokenizer(
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

        encoded = self.tokenizer.tokenizer(
            paragraphs,
            target_clauses,
            truncation=True,
            padding=True,
            max_length=self.tokenizer.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        encoded["labels"] = labels

        return encoded