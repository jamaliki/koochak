#!/usr/bin/env python3
"""Recluster immutable Progres score matrices without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.progres_clustering import endpoint_summary, summarize_scores


SAME_FOLD_THRESHOLD = 0.8


def _labels(sample_count: int) -> list[str]:
    return [f"sample_{index:05d}" for index in range(sample_count)]


def _section(
    source: dict[str, object], name: str, labels: list[str], designable: set[str],
) -> dict[str, object]:
    matrix = np.asarray(source[name]["score_matrix"], dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"{name} score matrix shape does not match sample count")
    indices = [index for index, label in enumerate(labels) if label in designable]
    return {
        "all_samples": summarize_scores(
            labels, matrix, range(len(labels)), threshold=SAME_FOLD_THRESHOLD,
            population="all_samples", embedding_view=f"{name} backbone embeddings",
        ),
        "designable_subset": summarize_scores(
            labels, matrix, indices, threshold=SAME_FOLD_THRESHOLD,
            population="ESMFold-designable sample IDs",
            embedding_view=f"{name} backbone embeddings",
        ),
        "score_matrix": matrix.tolist(),
    }


def recluster(source: dict[str, object], *, source_path: Path, label: str | None = None) -> dict[str, object]:
    """Return a complete-linkage replacement for one stored analysis JSON."""

    if source.get("same_fold_threshold", SAME_FOLD_THRESHOLD) != SAME_FOLD_THRESHOLD:
        raise ValueError("stored Progres threshold is not the established 0.8 threshold")
    generated_matrix = np.asarray(source["generated"]["score_matrix"], dtype=float)
    esmfold_matrix = np.asarray(source["esmfold"]["score_matrix"], dtype=float)
    if generated_matrix.shape != esmfold_matrix.shape or generated_matrix.ndim != 2:
        raise ValueError("generated and ESMFold score matrices must have the same rank-2 shape")
    sample_count = generated_matrix.shape[0]
    labels = _labels(sample_count)
    designable_samples = {str(item) for item in source.get("designable_samples", [])}
    if not designable_samples <= set(labels):
        raise ValueError("designable sample IDs do not match the stored matrix labels")
    result = {
        key: value for key, value in source.items() if key not in {"generated", "esmfold"}
    }
    result.update(
        {
            "method": "Progres v1.1.0 complete-linkage reanalysis of immutable score matrices",
            "cluster_linkage": "complete linkage using minimum within-cluster Progres similarity at >= 0.8",
            "generated": _section(source, "generated", labels, designable_samples),
            "esmfold": _section(source, "esmfold", labels, designable_samples),
            "reclustered_from": str(source_path.resolve()),
            "source_input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    )
    result.update(endpoint_summary(result["generated"], result["esmfold"], len(designable_samples)))
    if label is not None:
        result["panel"] = label
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = recluster(source, source_path=args.input, label=args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "primary_designable_progres_cluster_count": result["primary_designable_progres_cluster_count"],
        "secondary_esmfold_designable_progres_cluster_count": result["secondary_esmfold_designable_progres_cluster_count"],
        "designable_progres_cluster_count_disagreement": result["designable_progres_cluster_count_disagreement"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
