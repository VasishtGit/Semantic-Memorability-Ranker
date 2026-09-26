"""Load and prepare the Georgiou human-memory evaluation dataset.

This module loads human-rated memorability records from the Georgiou benchmark,
groups clauses by their originating story, sorts clauses by position to reconstruct
the full narrative paragraph, and provides structured evaluation objects.
"""

from __future__ import annotations

import json
from pathlib import Path


# File path to the raw Georgiou human memorability dataset
GEORGIOU_DATASET = Path(
    "data/raw/human_memorability_dataset.jsonl"
)

def load_human_memorability():
    """Load the human-rated clauses from the Georgiou dataset.

    Returns:
        List of parsed JSON dictionaries representing individual clause records.
    """
    rows = []

    # Read records from the JSONL file
    with open(
        GEORGIOU_DATASET,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def prepare_georgiou_dataset():
    """Group human-rated clauses into complete stories.

    Reconstructs full narrative paragraphs by concatenating clauses in clause_id order.

    Returns:
        List of dictionaries, each containing 'story', 'paragraph', and 'clauses'.
    """
    rows = load_human_memorability()

    stories = {}

    # Group individual clause records by story title
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

    # Format each story and reconstruct narrative paragraph text
    for story_data in stories.values():
        # Sort clauses in canonical sequential narrative order
        story_data["clauses"].sort(
            key=lambda clause: clause["clause_id"]
        )

        # Reconstruct narrative paragraph by joining ordered clause texts
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


# Dataset inspection and summary when executed directly
if __name__ == "__main__":
    dataset = prepare_georgiou_dataset()

    print(f"Stories: {len(dataset)}")

    total_clauses = sum(
        len(story["clauses"])
        for story in dataset
    )

    print(f"Clauses: {total_clauses}")

    # Print clause counts per story
    for story in dataset:
        print(
            f"{story['story']}: "
            f"{len(story['clauses'])} clauses"
        )

    # Print sample data from the first story
    first_story = dataset[0]

    print("\nFirst story:")
    print("Story:", first_story["story"])
    print("Paragraph:", first_story["paragraph"][:300])
    print("First clause:", first_story["clauses"][0])
