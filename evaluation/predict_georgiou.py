"""Generate memorability predictions on the Georgiou benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset.tokenizer import MemorabilityTokenizer
from evaluation.georgiou_dataset import prepare_georgiou_dataset
from models.memorability_ranker import SemanticMemorabilityRanker


CHECKPOINT_FILE = Path(
    "checkpoints/best.pt"
)

OUTPUT_FILE = Path(
    "evaluation/georgiou_predictions.json"
)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # Load the trained model.
    model = SemanticMemorabilityRanker(
        unfreeze_last_n_layers=2,
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint: {CHECKPOINT_FILE}"
    )

    # Load and prepare the Georgiou dataset.
    stories = prepare_georgiou_dataset()

    total_clauses = sum(
        len(story["clauses"])
        for story in stories
    )

    print(
        f"Stories: {len(stories)}"
    )

    print(
        f"Clauses: {total_clauses}"
    )

    tokenizer = MemorabilityTokenizer()

    predictions = []

    processed = 0

    with torch.no_grad():

        for story in stories:

            paragraph = story["paragraph"]

            for clause in story["clauses"]:

                encoded = tokenizer(
                    paragraph,
                    clause["text"],
                )

                input_ids = encoded[
                    "input_ids"
                ].unsqueeze(0).to(device)

                attention_mask = encoded[
                    "attention_mask"
                ].unsqueeze(0).to(device)

                clause_mask = encoded[
                    "clause_mask"
                ].unsqueeze(0).to(device)

                prediction = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    clause_mask=clause_mask,
                )

                predicted_memorability = float(
                    prediction.item()
                )

                predictions.append(
                    {
                        "story": story["story"],
                        "clause_id": clause["clause_id"],
                        "text": clause["text"],
                        "human_memorability": float(
                            clause["memorability"]
                        ),
                        "predicted_memorability": predicted_memorability,
                    }
                )

                processed += 1

                if processed % 50 == 0:
                    print(
                        f"Processed "
                        f"{processed}/{total_clauses}"
                    )

    result = {
        "checkpoint": str(
            CHECKPOINT_FILE
        ),
        "stories": len(stories),
        "clauses": len(predictions),
        "predictions": predictions,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()

    print(
        f"Saved predictions to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Stories: {result['stories']}"
    )

    print(
        f"Clauses: {result['clauses']}"
    )


if __name__ == "__main__":
    main()