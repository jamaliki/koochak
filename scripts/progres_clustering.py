"""Deterministic Progres similarity clustering and summary statistics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np


def complete_linkage_clusters(scores: np.ndarray, threshold: float) -> list[list[int]]:
    """Cluster scores without transitive threshold chaining.

    Similarity is used directly, so complete linkage between two groups is the
    minimum cross-group similarity.  The highest-linkage merge is selected at
    each step, and ties are resolved by the lexicographically smallest merged
    member list.  A merge is allowed only when every pair in the merged group
    remains at or above ``threshold``.
    """

    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Progres scores must be a square matrix")
    if not np.isfinite(values).all():
        raise ValueError("Progres scores must be finite")
    if not np.allclose(values, values.T, rtol=0.0, atol=1e-6):
        raise ValueError("Progres scores must be symmetric")
    if not math.isfinite(threshold):
        raise ValueError("Progres threshold must be finite")

    groups = [[index] for index in range(len(values))]
    while True:
        best: tuple[float, tuple[int, ...], int, int] | None = None
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                cross = values[np.ix_(groups[left], groups[right])]
                linkage = float(cross.min())
                merged = tuple(sorted((*groups[left], *groups[right])))
                candidate = (linkage, merged, left, right)
                if linkage < threshold:
                    continue
                if best is None or linkage > best[0] or (
                    linkage == best[0] and merged < best[1]
                ):
                    best = candidate
        if best is None:
            break
        _linkage, merged, left, right = best
        groups = [
            group
            for index, group in enumerate(groups)
            if index not in (left, right)
        ]
        groups.append(list(merged))
    return sorted(groups, key=lambda group: (-len(group), group))


def effective_cluster_count(groups: list[list[int]]) -> float:
    """Return the exponential entropy of the cluster-size distribution."""

    total = sum(map(len, groups))
    if not total:
        return 0.0
    probabilities = [len(group) / total for group in groups]
    return math.exp(
        -sum(probability * math.log(probability) for probability in probabilities)
    )


def _within_group_values(scores: np.ndarray, group: list[int]) -> list[float]:
    return [
        float(scores[left, right])
        for offset, left in enumerate(group)
        for right in group[offset + 1 :]
    ]


def _medoid(scores: np.ndarray, group: list[int]) -> int:
    if len(group) == 1:
        return group[0]
    ranked = []
    for candidate in group:
        similarities = [
            float(scores[candidate, other]) for other in group if other != candidate
        ]
        ranked.append((statistics.fmean(similarities), -candidate, candidate))
    return max(ranked)[2]


def summarize_scores(
    labels: Sequence[str],
    scores: np.ndarray,
    indices: Sequence[int],
    *,
    threshold: float,
    population: str,
    embedding_view: str,
) -> dict[str, object]:
    """Summarize one population using complete-linkage Progres clusters."""

    values = np.asarray(scores, dtype=float)
    selected = [int(index) for index in indices]
    if len(labels) != len(values):
        raise ValueError("label count does not match Progres score matrix")
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= len(labels) for index in selected
    ):
        raise ValueError("Progres subset indices must be unique and in range")

    subset = values[np.ix_(selected, selected)]
    pairs = [
        float(subset[left, right])
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    ]
    groups = complete_linkage_clusters(subset, threshold)
    cluster_labels = [[str(labels[selected[index]]) for index in group] for group in groups]
    medoid_indices = [_medoid(subset, group) for group in groups]
    cluster_medoids = [str(labels[selected[index]]) for index in medoid_indices]
    cluster_details = []
    for group, medoid in zip(cluster_labels, cluster_medoids, strict=True):
        local_group = groups[len(cluster_details)]
        within = _within_group_values(subset, local_group)
        cluster_details.append(
            {
                "medoid": medoid,
                "representative": medoid,
                "members": group,
                "size": len(group),
                "minimum_pair_similarity": min(within) if within else 1.0,
                "mean_pair_similarity": statistics.fmean(within) if within else 1.0,
            }
        )
    same_fold_pair_count = sum(
        score >= threshold for score in pairs
    )
    same_cluster_pair_count = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    return {
        "population": population,
        "embedding_view": embedding_view,
        "sample_count": len(selected),
        "pair_count": len(pairs),
        "pair_similarity_mean": statistics.fmean(pairs) if pairs else None,
        "pair_similarity_median": statistics.median(pairs) if pairs else None,
        "same_fold_pair_count": same_fold_pair_count,
        "same_fold_pair_fraction": same_fold_pair_count / len(pairs) if pairs else None,
        "same_cluster_pair_count": same_cluster_pair_count,
        "same_cluster_pair_fraction": (
            same_cluster_pair_count / len(pairs) if pairs else None
        ),
        "nearest_neighbor_similarity_mean": statistics.fmean(
            [
                max(
                    float(subset[index, other])
                    for other in range(len(selected))
                    if other != index
                )
                for index in range(len(selected))
                if len(selected) > 1
            ]
        ) if len(selected) > 1 else None,
        "cluster_count": len(groups),
        "effective_cluster_count": effective_cluster_count(groups),
        "largest_cluster_size": len(groups[0]) if groups else 0,
        "largest_cluster_fraction": len(groups[0]) / len(selected) if groups else None,
        "cluster_sizes": [len(group) for group in groups],
        "clusters": cluster_labels,
        "cluster_medoids": cluster_medoids,
        "representatives": cluster_medoids,
        "cluster_details": cluster_details,
        "cluster_threshold": threshold,
        "cluster_linkage": (
            "complete linkage using minimum within-cluster Progres similarity "
            f"at >= {threshold:g}"
        ),
    }


def endpoint_summary(
    generated: dict[str, object],
    esmfold: dict[str, object],
    designable_count: int,
) -> dict[str, object]:
    """Name the generated primary endpoint and refolded concordance endpoint."""

    generated_subset = generated["designable_subset"]
    esmfold_subset = esmfold["designable_subset"]
    generated_count = int(generated_subset["cluster_count"])
    esmfold_count = int(esmfold_subset["cluster_count"])
    disagreement = generated_count != esmfold_count
    return {
        "primary_metric": {
            "name": "generated.designable_subset.complete_linkage_progres_cluster_count",
            "value": generated_count,
            "population": "ESMFold-designable sample IDs",
            "embedding_view": "generated backbone embeddings",
            "designable_count": designable_count,
        },
        "secondary_concordance_metric": {
            "name": "esmfold.designable_subset.complete_linkage_progres_cluster_count",
            "value": esmfold_count,
            "population": "ESMFold-designable sample IDs",
            "embedding_view": "ESMFold-refolded backbone embeddings",
        },
        "primary_designable_progres_cluster_count": generated_count,
        "secondary_esmfold_designable_progres_cluster_count": esmfold_count,
        "designable_progres_cluster_count": generated_count,
        "designable_progres_cluster_count_disagreement": disagreement,
        "designable_progres_cluster_count_delta_generated_minus_esmfold": (
            generated_count - esmfold_count
        ),
    }
