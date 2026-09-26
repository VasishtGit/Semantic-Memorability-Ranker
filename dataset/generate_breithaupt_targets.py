"""Generate clause-level Breithaupt memorability targets.

This script processes segmented narrative stories and their multi-generation retellings
from the Breithaupt dataset. For each clause in an original story, it identifies the
top semantic candidate clauses in each retelling generation using dense bi-encoder
embeddings, scores semantic preservation with a MeaningBERT cross-encoder, combines
generation scores using transmission chain weights (0.2, 0.3, 0.5), and normalizes
the resulting memorability scores to within-story percentile ranks.
"""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# Path to input segmented stories containing original and retelling clauses
INPUT_PATH = Path(
    "data/processed/breithaupt_segmented.json"
)

# Output destination for computed clause memorability targets
OUTPUT_PATH = Path(
    "data/processed/breithaupt_targets.json"
)

# The paper specifies 384-dimensional semantic embeddings,
# but does not identify the exact embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MeaningBERT is used as the semantic matching scorer.
MEANINGBERT_MODEL = "davebulaval/MeaningBERT"

# Number of top candidate clauses to retrieve per retelling generation via cosine similarity
TOP_K = 5


def cosine_top_k(
    query_embedding,
    candidate_embeddings,
    k=TOP_K,
):
    """Retrieve the top-k candidate clause embeddings closest to the query embedding.

    Args:
        query_embedding: Normalized query vector of shape (D,).
        candidate_embeddings: Matrix of candidate vectors of shape (N, D).
        k: Maximum number of top candidates to retrieve.

    Returns:
        Tuple of (top_similarity_values, top_indices).
    """
    # Compute cosine similarities between the single query vector and all candidates
    scores = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),
        candidate_embeddings,
        dim=1,
    )

    # Bound k by the number of candidate clauses available
    k = min(
        k,
        len(scores),
    )

    # Extract top-k highest similarity scores and corresponding indices
    values, indices = torch.topk(
        scores,
        k=k,
    )

    return values, indices


def percentile_normalize(
    scores,
):
    """Convert raw scores to within-story percentile ranks.

    Tied scores receive the average rank of the tied group. The resulting ranks
    are normalized linearly to the range [0.0, 1.0].

    Args:
        scores: Sequence of raw numerical scores.

    Returns:
        List of percentile-normalized float scores in the interval [0.0, 1.0].
    """

    n = len(scores)

    # Return default mid-point if there is only one score
    if n <= 1:
        return [0.5] * n

    # Return mid-point values if all scores are identical
    if all(
        score == scores[0]
        for score in scores
    ):
        return [0.5] * n

    # Obtain indices sorted by score in ascending order
    sorted_indices = sorted(
        range(n),
        key=lambda i: scores[i],
    )

    ranks = [0.0] * n

    position = 0

    # Assign fractional ranks handling tied values
    while position < n:
        end = position + 1

        # Identify contiguous span of identical scores
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

        # Compute mid-rank for the group of tied elements (1-based ranking)
        average_rank = (
            position
            + 1
            + end
        ) / 2.0

        # Assign computed average rank to each tied position
        for j in range(
            position,
            end,
        ):
            ranks[
                sorted_indices[j]
            ] = average_rank

        position = end

    # Normalize ranks into [0.0, 1.0] range
    normalized = [
        (rank - 1.0) / (n - 1.0)
        for rank in ranks
    ]

    return normalized


def main():
    """Execute the end-to-end Breithaupt memorability target generation pipeline."""
    # Load input segmented stories and retellings from JSON
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        stories = json.load(f)

    # Determine runtime hardware device
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

    # Initialize the sentence transformer for candidate retrieval
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )

    # Initialize the cross-encoder tokenizer and model for semantic matching
    meaning_tokenizer = AutoTokenizer.from_pretrained(
        MEANINGBERT_MODEL,
    )

    meaning_model = AutoModelForSequenceClassification.from_pretrained(
        MEANINGBERT_MODEL,
    ).to(device)

    meaning_model.eval()

    all_outputs = []

    # Process each story and its associated retelling generations
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

        # Compute normalized bi-encoder embeddings for original clauses
        original_embeddings = embedding_model.encode(
            original_clauses,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Compute normalized bi-encoder embeddings for each retelling generation
        retelling_embeddings = [
            embedding_model.encode(
                clauses,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            for clauses in retellings
        ]

        results = []

        # Iterate over each clause in the original narrative
        for i, original_clause in enumerate(
            original_clauses
        ):
            retelling_scores = []

            # Retrieve top candidates and score semantic preservation for each generation
            for g in range(3):
                # Retrieve top-k nearest candidate clauses by cosine similarity
                _, candidate_indices = cosine_top_k(
                    original_embeddings[i],
                    retelling_embeddings[g],
                )

                candidates = [
                    retellings[g][index]
                    for index in candidate_indices.tolist()
                ]

                # Format original-clause and candidate pairs for cross-encoder scoring
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

                # Predict semantic preservation logits with MeaningBERT
                with torch.no_grad():
                    logits = meaning_model(
                        **inputs
                    ).logits

                scores = logits.squeeze(-1)

                # Select highest scoring semantic match among the top candidates
                best_index = torch.argmax(
                    scores
                ).item()

                best_score = scores[
                    best_index
                ].item()

                retelling_scores.append(
                    best_score
                )

            # Compute weighted composite memorability score across generations
            # Retelling 3 carries the highest weight as it reflects longest survival
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

        # Extract raw scores for within-story percentile normalization
        raw_scores = [
            result["memorability_raw"]
            for result in results
        ]

        normalized_scores = percentile_normalize(
            raw_scores
        )

        # Assign normalized percentile scores to each clause record
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

    # Ensure output destination directory exists
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Persist structured targets to disk
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
