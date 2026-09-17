"""Load and prepare the Georgiou human-memory evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path


GEORGIOU_DATASET = Path(
    "data/raw/human_memorability_dataset.jsonl"
)

def load_human_memorability():
    """Load the human-rated clauses from the Georgiou dataset."""
    rows = []

    with open(
        GEORGIOU_DATASET,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def prepare_georgiou_dataset():
    """Group human-rated clauses into complete stories."""
    rows = load_human_memorability()

    stories = {}

    for row in rows:
        story = row["story"]

        if story not in stories:
            stories[story] = {
                "story": story,
                "clauses": [],
            }

        stories[story]["clauses"].append(
            {
                "clause_id": row["clause_id"],
                "text": row["text"],
                "memorability": float(row["memorability"]),
            }
        )

    dataset = []

    for story_data in stories.values():
        story_data["clauses"].sort(
            key=lambda clause: clause["clause_id"]
        )

        paragraph = " ".join(
            clause["text"]
            for clause in story_data["clauses"]
        )

        dataset.append(
            {
                "story": story_data["story"],
                "paragraph": paragraph,
                "clauses": story_data["clauses"],
            }
        )

    return dataset


if __name__ == "__main__":
    dataset = prepare_georgiou_dataset()

    print(f"Stories: {len(dataset)}")

    total_clauses = sum(
        len(story["clauses"])
        for story in dataset
    )

    print(f"Clauses: {total_clauses}")

    for story in dataset:
        print(
            f"{story['story']}: "
            f"{len(story['clauses'])} clauses"
        )

    first_story = dataset[0]

    print("\nFirst story:")
    print("Story:", first_story["story"])
    print("Paragraph:", first_story["paragraph"][:300])
    print("First clause:", first_story["clauses"][0])