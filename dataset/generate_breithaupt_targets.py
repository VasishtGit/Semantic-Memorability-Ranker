"""Generate clause-level Breithaupt memorability targets."""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


INPUT_PATH = Path(
    "data/processed/breithaupt_segmented.json"
)

OUTPUT_PATH = Path(
    "data/processed/breithaupt_targets.json"
)

# The paper specifies 384-dimensional semantic embeddings,
# but does not identify the exact embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MeaningBERT is used as the semantic matching scorer.
MEANINGBERT_MODEL = "davebulaval/MeaningBERT"

TOP_K = 5


def cosine_top_k(
    query_embedding,
    candidate_embeddings,
    k=TOP_K,
):
    scores = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),
        candidate_embeddings,
        dim=1,
    )

    k = min(
        k,
        len(scores),
    )

    values, indices = torch.topk(
        scores,
        k=k,
    )

    return values, indices


def percentile_normalize(
    scores,
):
    """Convert raw scores to within-story percentile ranks."""

    n = len(scores)

    if n <= 1:
        return [0.5] * n

    if all(
        score == scores[0]
        for score in scores
    ):
        return [0.5] * n

    sorted_indices = sorted(
        range(n),
        key=lambda i: scores[i],
    )

    ranks = [0.0] * n

    position = 0

    while position < n:
        end = position + 1

        while (
            end < n
            and scores[
                sorted_indices[end]
            ]
            == scores[
                sorted_indices[position]
            ]
        ):
            end += 1

        average_rank = (
            position
            + 1
            + end
        ) / 2.0

        for j in range(
            position,
            end,
        ):
            ranks[
                sorted_indices[j]
            ] = average_rank

        position = end

    normalized = [
        (rank - 1.0) / (n - 1.0)
        for rank in ranks
    ]

    return normalized


def main():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        stories = json.load(f)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )
    print(
        f"MeaningBERT model: "
        f"{MEANINGBERT_MODEL}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )

    meaning_tokenizer = AutoTokenizer.from_pretrained(
        MEANINGBERT_MODEL,
    )

    meaning_model = AutoModelForSequenceClassification.from_pretrained(
        MEANINGBERT_MODEL,
    ).to(device)

    meaning_model.eval()

    all_outputs = []

    for story_number, story in enumerate(
        stories,
        start=1,
    ):
        original_clauses = story[
            "original_clauses"
        ]

        retellings = [
            story["retelling_1_clauses"],
            story["retelling_2_clauses"],
            story["retelling_3_clauses"],
        ]

        print()
        print(
            f"Story {story_number}/{len(stories)}"
        )
        print(
            f"Story ID: "
            f"{story['story_id']}"
        )
        print(
            f"Original clauses: "
            f"{len(original_clauses)}"
        )

        original_embeddings = embedding_model.encode(
            original_clauses,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        retelling_embeddings = [
            embedding_model.encode(
                clauses,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            for clauses in retellings
        ]

        results = []

        for i, original_clause in enumerate(
            original_clauses
        ):
            retelling_scores = []

            for g in range(3):
                _, candidate_indices = cosine_top_k(
                    original_embeddings[i],
                    retelling_embeddings[g],
                )

                candidates = [
                    retellings[g][index]
                    for index in candidate_indices.tolist()
                ]

                inputs = meaning_tokenizer(
                    [original_clause] * len(candidates),
                    candidates,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():
                    logits = meaning_model(
                        **inputs
                    ).logits

                scores = logits.squeeze(-1)

                best_index = torch.argmax(
                    scores
                ).item()

                best_score = scores[
                    best_index
                ].item()

                retelling_scores.append(
                    best_score
                )

            memorability = (
                0.2 * retelling_scores[0]
                + 0.3 * retelling_scores[1]
                + 0.5 * retelling_scores[2]
            )

            results.append(
                {
                    "clause_index": i,
                    "clause": original_clause,
                    "retelling_scores": retelling_scores,
                    "memorability_raw": memorability,
                }
            )

        raw_scores = [
            result["memorability_raw"]
            for result in results
        ]

        normalized_scores = percentile_normalize(
            raw_scores
        )

        for result, normalized_score in zip(
            results,
            normalized_scores,
        ):
            result[
                "memorability"
            ] = normalized_score

        all_outputs.append(
            {
                "story_id": story["story_id"],
                "clauses": results,
            }
        )

        print(
            f"Raw score range: "
            f"{min(raw_scores):.4f} - "
            f"{max(raw_scores):.4f}"
        )

        print(
            f"Normalized range: "
            f"{min(normalized_scores):.4f} - "
            f"{max(normalized_scores):.4f}"
        )

    output = {
        "stories": all_outputs,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    total_clauses = sum(
        len(story["clauses"])
        for story in all_outputs
    )

    print()
    print(
        f"Stories processed: "
        f"{len(all_outputs)}"
    )
    print(
        f"Total clauses: "
        f"{total_clauses}"
    )
    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()