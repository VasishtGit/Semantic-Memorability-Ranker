"""Evaluate Georgiou benchmark predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


PREDICTIONS_FILE = Path(
    "evaluation/georgiou_predictions.json"
)

AUDIT_FILE = Path(
    "evaluation/georgiou_metric_audit.json"
)


def dcg(
    relevances,
):
    """Calculate discounted cumulative gain."""
    relevances = np.asarray(
        relevances,
        dtype=float,
    )

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(
            2,
            len(relevances) + 2,
        )
    )

    return float(
        np.sum(
            (2.0 ** relevances - 1.0)
            / discounts
        )
    )


def ndcg_for_story(
    predictions,
    targets,
    k,
):
    """Calculate NDCG@k for one story."""
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    if len(targets) == 0:
        return 0.0

    k = min(
        k,
        len(targets),
    )

    predicted_order = np.argsort(
        -predictions,
        kind="stable",
    )[:k]

    ideal_order = np.argsort(
        -targets,
        kind="stable",
    )[:k]

    predicted_dcg = dcg(
        targets[predicted_order]
    )

    ideal_dcg = dcg(
        targets[ideal_order]
    )

    if ideal_dcg == 0:
        return 0.0

    return float(
        predicted_dcg / ideal_dcg
    )


def top_k_overlap_for_story(
    predictions,
    targets,
    k=5,
):
    """Calculate top-k set overlap for one story."""
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    if len(targets) == 0:
        return 0.0

    k = min(
        k,
        len(targets),
    )

    predicted_top = set(
        np.argsort(
            -predictions,
            kind="stable",
        )[:k]
    )

    actual_top = set(
        np.argsort(
            -targets,
            kind="stable",
        )[:k]
    )

    return float(
        len(
            predicted_top & actual_top
        )
        / k
    )


def pairwise_accuracy_for_story(
    predictions,
    targets,
):
    """Calculate pairwise ranking accuracy for one story."""
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    correct = 0
    total = 0

    for i in range(
        len(targets)
    ):
        for j in range(
            i + 1,
            len(targets),
        ):
            target_difference = (
                targets[i]
                - targets[j]
            )

            # Ignore human-score ties.
            if target_difference == 0:
                continue

            prediction_difference = (
                predictions[i]
                - predictions[j]
            )

            if (
                np.sign(
                    prediction_difference
                )
                == np.sign(
                    target_difference
                )
            ):
                correct += 1

            total += 1

    if total == 0:
        return 0.0

    return float(
        correct / total
    )


def safe_rank_correlation(
    statistic,
):
    """Convert a scipy correlation result to a finite float when possible."""
    if statistic is None:
        return 0.0

    value = float(statistic)

    if not np.isfinite(value):
        return 0.0

    return value


def evaluate_story(
    predictions,
    targets,
):
    """Calculate all ranking metrics for one story."""
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    if len(predictions) != len(targets):
        raise ValueError(
            "Predictions and targets must "
            "contain the same number of clauses."
        )

    if len(targets) == 0:
        raise ValueError(
            "Cannot evaluate an empty story."
        )

    spearman = spearmanr(
        predictions,
        targets,
    ).statistic

    kendall = kendalltau(
        predictions,
        targets,
    ).statistic

    return {
        "clauses": len(targets),

        "mae": float(
            mean_absolute_error(
                targets,
                predictions,
            )
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    targets,
                    predictions,
                )
            )
        ),

        "spearman": safe_rank_correlation(
            spearman
        ),

        "kendall_tau": safe_rank_correlation(
            kendall
        ),

        "pairwise_accuracy": (
            pairwise_accuracy_for_story(
                predictions,
                targets,
            )
        ),

        "ndcg_at_1": (
            ndcg_for_story(
                predictions,
                targets,
                1,
            )
        ),

        "ndcg_at_3": (
            ndcg_for_story(
                predictions,
                targets,
                3,
            )
        ),

        "ndcg_at_5": (
            ndcg_for_story(
                predictions,
                targets,
                5,
            )
        ),

        "top_5_overlap": (
            top_k_overlap_for_story(
                predictions,
                targets,
                5,
            )
        ),
    }


def evaluate_dataset(
    rows,
):
    """Calculate aggregate and per-story metrics."""
    if not rows:
        raise ValueError(
            "Prediction file contains no rows."
        )

    stories = {}

    for row in rows:
        required_fields = {
            "story",
            "predicted_memorability",
            "human_memorability",
        }

        missing_fields = (
            required_fields
            - row.keys()
        )

        if missing_fields:
            raise ValueError(
                "Prediction row is missing "
                f"required fields: "
                f"{sorted(missing_fields)}"
            )

        story = row["story"]

        stories.setdefault(
            story,
            {
                "predictions": [],
                "targets": [],
            },
        )

        stories[story][
            "predictions"
        ].append(
            float(
                row["predicted_memorability"]
            )
        )

        stories[story][
            "targets"
        ].append(
            float(
                row["human_memorability"]
            )
        )

    per_story = {}

    for story in sorted(stories):
        predictions = stories[
            story
        ]["predictions"]

        targets = stories[
            story
        ]["targets"]

        per_story[story] = evaluate_story(
            predictions,
            targets,
        )

    all_predictions = [
        float(
            row["predicted_memorability"]
        )
        for row in rows
    ]

    all_targets = [
        float(
            row["human_memorability"]
        )
        for row in rows
    ]

    aggregate_spearman = spearmanr(
        all_predictions,
        all_targets,
    ).statistic

    aggregate_kendall = kendalltau(
        all_predictions,
        all_targets,
    ).statistic

    # Pairwise accuracy is computed only within
    # individual stories. Cross-story comparisons
    # are not meaningful for this benchmark.
    aggregate_pairwise_correct = 0
    aggregate_pairwise_total = 0

    for story in sorted(stories):
        predictions = np.asarray(
            stories[story]["predictions"],
            dtype=float,
        )

        targets = np.asarray(
            stories[story]["targets"],
            dtype=float,
        )

        for i in range(
            len(targets)
        ):
            for j in range(
                i + 1,
                len(targets),
            ):
                target_difference = (
                    targets[i]
                    - targets[j]
                )

                if target_difference == 0:
                    continue

                prediction_difference = (
                    predictions[i]
                    - predictions[j]
                )

                if (
                    np.sign(
                        prediction_difference
                    )
                    == np.sign(
                        target_difference
                    )
                ):
                    aggregate_pairwise_correct += 1

                aggregate_pairwise_total += 1

    if aggregate_pairwise_total == 0:
        aggregate_pairwise_accuracy = 0.0
    else:
        aggregate_pairwise_accuracy = (
            aggregate_pairwise_correct
            / aggregate_pairwise_total
        )

    aggregate_ndcg = {}

    for k in (
        1,
        3,
        5,
    ):
        aggregate_ndcg[
            f"ndcg_at_{k}"
        ] = float(
            np.mean(
                [
                    per_story[story][
                        f"ndcg_at_{k}"
                    ]
                    for story in per_story
                ]
            )
        )

    aggregate_top5 = float(
        np.mean(
            [
                per_story[story][
                    "top_5_overlap"
                ]
                for story in per_story
            ]
        )
    )

    aggregate = {
        "stories": len(stories),
        "clauses": len(rows),

        "mae": float(
            mean_absolute_error(
                all_targets,
                all_predictions,
            )
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    all_targets,
                    all_predictions,
                )
            )
        ),

        "spearman": safe_rank_correlation(
            aggregate_spearman
        ),

        "kendall_tau": safe_rank_correlation(
            aggregate_kendall
        ),

        "pairwise_accuracy": float(
            aggregate_pairwise_accuracy
        ),

        **aggregate_ndcg,

        "top_5_overlap": aggregate_top5,
    }

    return {
        "aggregate": aggregate,
        "per_story": per_story,
    }


def main():
    """Load predictions, evaluate them, and save the audit."""
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions not found: "
            f"{PREDICTIONS_FILE}"
        )

    with open(
        PREDICTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)

    if "predictions" not in result:
        raise ValueError(
            "Prediction file does not contain "
            "a 'predictions' field."
        )

    rows = result["predictions"]

    audit = evaluate_dataset(
        rows
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        AUDIT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
        )

    aggregate = audit[
        "aggregate"
    ]

    print()
    print(
        "GEORGIOU METRIC AUDIT"
    )
    print(
        "====================="
    )

    for name, value in aggregate.items():
        if isinstance(
            value,
            float,
        ):
            print(
                f"{name:20s}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{name:20s}: "
                f"{value}"
            )

    print()
    print(
        "PER-STORY RESULTS"
    )
    print(
        "================="
    )

    for story, metrics in audit[
        "per_story"
    ].items():

        print()
        print(
            f"{story} "
            f"({metrics['clauses']} clauses)"
        )

        print(
            f"  MAE             : "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"  RMSE            : "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"  Spearman        : "
            f"{metrics['spearman']:.4f}"
        )

        print(
            f"  Kendall tau     : "
            f"{metrics['kendall_tau']:.4f}"
        )

        print(
            f"  Pairwise Acc.   : "
            f"{metrics['pairwise_accuracy']:.4f}"
        )

        print(
            f"  NDCG@1          : "
            f"{metrics['ndcg_at_1']:.4f}"
        )

        print(
            f"  NDCG@3          : "
            f"{metrics['ndcg_at_3']:.4f}"
        )

        print(
            f"  NDCG@5          : "
            f"{metrics['ndcg_at_5']:.4f}"
        )

        print(
            f"  Top-5 overlap   : "
            f"{metrics['top_5_overlap']:.4f}"
        )

    print()
    print(
        f"Audit saved to: "
        f"{AUDIT_FILE}"
    )


if __name__ == "__main__":
    main()