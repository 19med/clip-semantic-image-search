"""Shared metrics for all retrieval systems."""

from __future__ import annotations

import statistics


def evaluate_retrieval(results: list[dict], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    if not results:
        raise ValueError("Cannot evaluate an empty result set.")
    recalls = {k: 0 for k in ks}
    reciprocal_ranks = []
    ranks = []

    for result in results:
        target = int(result["ground_truth_image_id"])
        ranked = [int(image_id) for image_id in result["ranked_image_ids"]]
        rank = next((index + 1 for index,
                    image_id in enumerate(ranked) if image_id == target), None)
        for k in ks:
            recalls[k] += int(rank is not None and rank <= k)
        if rank is not None:
            ranks.append(rank)
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(results)
    metrics = {f"recall_at_{k}": recalls[k] / total for k in ks}
    metrics["mean_reciprocal_rank"] = sum(reciprocal_ranks) / total
    metrics["median_rank_found"] = statistics.median(ranks) if ranks else None
    metrics["num_queries"] = total
    return metrics
