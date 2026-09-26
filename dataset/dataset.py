"""Dataset wrapper for loading memorability training examples from JSONL files.

This module defines a PyTorch Dataset that ingests JSON Lines (JSONL) records containing
narrative text, candidate clauses, story identifiers, and empirical memorability scores.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class MemorabilityDataset(Dataset):
    """JSONL-backed dataset for paragraph/clause memorability pairs."""

    def __init__(
        self,
        jsonl_path: str | Path,
    ):
        """Load and parse memorability examples from a JSONL file.

        Args:
            jsonl_path: File system path to the JSONL dataset file.
        """
        # Internal storage for loaded sample records
        self.samples = []

        jsonl_path = Path(jsonl_path)

        # Open and iterate through lines in the JSONL file
        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Parse JSON record
                sample = json.loads(line)

                # Standardize sample dictionary with consistent keys
                self.samples.append(
                    {
                        # Support either narrative_id or story_id key
                        "narrative_id": sample.get(
                            "narrative_id",
                            sample.get("story_id"),
                        ),
                        "paragraph": sample["paragraph"],
                        "target_clause": sample["target_clause"],
                        # Convert target memorability score to float
                        "label": float(
                            sample["memorability"]
                        ),
                    }
                )

    def __len__(self):
        """Return the total number of samples loaded in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Retrieve a single memorability sample by index.

        Args:
            index: Integer index of the desired sample.

        Returns:
            Dictionary containing narrative_id, paragraph, target_clause, and label.
        """
        return self.samples[index]
