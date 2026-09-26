"""Prepare Breithaupt clause-level training data.

This script extracts complete original human narratives from the Breithaupt Excel workbook
and joins them with the generated clause-level memorability target scores, outputting a
standardized JSONL dataset suitable for model training.
"""

import json
from pathlib import Path

from openpyxl import load_workbook


# Path to precomputed clause memorability target scores
TARGETS_PATH = Path(
    "data/processed/breithaupt_targets.json"
)

# Path to the source Breithaupt archive Excel workbook containing original story texts
WORKBOOK_PATH = Path(
    r"C:\Datasets\jr2py-osfstorage-archive"
    r"\Stories and retellings"
    r"\All original stories and retellings.xlsx"
)

# Worksheet name in the workbook containing the original story pairs
SHEET_NAME = "Chat and Man"

# Destination path for the formatted training dataset
OUTPUT_PATH = Path(
    "data/processed/breithaupt_training.jsonl"
)


def load_original_stories():
    """Load original stories from the Breithaupt workbook.

    Returns:
        Dictionary mapping integer story IDs to their full narrative text strings.
    """
    # Load workbook in read-only and data-only mode for memory efficiency
    workbook = load_workbook(
        WORKBOOK_PATH,
        read_only=True,
        data_only=True,
    )

    # Select the target worksheet
    worksheet = workbook[
        SHEET_NAME
    ]

    stories = {}

    # Iterate through rows starting from row 3 (skipping header rows)
    for row in worksheet.iter_rows(
        min_row=3,
        values_only=True,
    ):
        # Column B = Original Story #
        story_id = row[1]

        # Column C = ORIGINAL STORY (human)
        original_story = row[2]

        # Skip rows where identifier or story content is missing
        if (
            story_id is None
            or original_story is None
        ):
            continue

        # Map integer story ID to full story text
        stories[int(story_id)] = str(
            original_story
        )

    # Close workbook resource
    workbook.close()

    return stories


def main():
    """Load targets and stories, combine into JSONL records, and save to disk."""
    # Load precomputed clause targets
    with open(
        TARGETS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        targets = json.load(f)

    # Load complete narrative paragraphs mapped by story ID
    original_stories = load_original_stories()

    total_examples = 0

    # Ensure output destination directory exists
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write training examples in JSON Lines format
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for story in targets["stories"]:
            story_id = int(
                story["story_id"]
            )

            # Ensure corresponding original narrative is present
            if story_id not in original_stories:
                raise ValueError(
                    f"Original story not found "
                    f"for story_id={story_id}"
                )

            paragraph = original_stories[
                story_id
            ]

            # Construct an individual training example for each clause
            for clause in story["clauses"]:
                example = {
                    "narrative_id": story_id,
                    "paragraph": paragraph,
                    "target_clause": clause[
                        "clause"
                    ],
                    "memorability": clause[
                        "memorability"
                    ],
                }

                # Write record as a single JSON line
                f.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_examples += 1

    print(
        f"Stories: "
        f"{len(targets['stories'])}"
    )

    print(
        f"Training examples: "
        f"{total_examples}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
