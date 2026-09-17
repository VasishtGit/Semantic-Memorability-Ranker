"""Dataset wrapper for loading memorability training examples from JSONL files."""

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
        self.samples = []

        jsonl_path = Path(jsonl_path)

        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                sample = json.loads(line)

                self.samples.append(
                    {
                        "narrative_id": sample.get(
                            "narrative_id",
                            sample.get("story_id"),
                        ),
                        "paragraph": sample["paragraph"],
                        "target_clause": sample["target_clause"],
                        "label": float(
                            sample["memorability"]
                        ),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]