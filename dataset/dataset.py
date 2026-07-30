"""Dataset wrapper for loading memorability training examples from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class NeuroDaptDataset(Dataset):
    """Simple JSONL-backed dataset for paragraph/clause memorability pairs."""

    def __init__(
        self,
        jsonl_path: str | Path,
    ):

        # Store parsed samples so they can be read by PyTorch DataLoader.
        self.samples = []

        # Resolve the path and read every non-empty line from the JSONL file.
        jsonl_path = Path(jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                # Convert each JSON line into a normalized sample dictionary.
                sample = json.loads(line)

                self.samples.append(
                    {
                        "paragraph": sample["paragraph"],
                        "target_clause": sample["target_clause"],
                        "label": float(sample["memorability"]),
                    }
                )

    def __len__(self):
        """Return the number of samples in the dataset."""

        return len(self.samples)

    def __getitem__(self, index):
        """Return a single sample by index."""

        return self.samples[index]