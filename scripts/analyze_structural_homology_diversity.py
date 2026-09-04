#!/usr/bin/env python3
"""Cluster generated proteins with structural-alignment TM scores."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import statistics

import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.tm import superimpose_structural_homologs, tm_score


def _load(file: Path):
    structure = PDBFile.read(file).get_structure(model=1)
    if not len(structure):
        raise ValueError(f"empty structure: {file}")
    return structure


def _pair_score(fixed, mobile) -> float:
    fitted, _transform, fixed_indices, mobile_indices = superimpose_structural_homologs(
        fixed, mobile, max_iterations=20, reference_length="shorter",
    )
    return float(tm_score(fixed, fitted, fixed_indices, mobile_indices, reference_length="shorter"))


def _clusters(scores: np.ndarray, threshold: float) -> list[list[int]]:
    unseen = set(range(len(scores)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        group = []
        while queue:
            current = queue.popleft()
            group.append(current)
            neighbors = {index for index in unseen if scores[current, index] >= threshold}
            unseen.difference_update(neighbors)
            queue.extend(sorted(neighbors))
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: (-len(group), group))


def _effective_cluster_count(groups: list[list[int]]) -> float:
    total = sum(map(len, groups))
    probabilities = [len(group) / total for group in groups]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return math.exp(entropy)


def analyze(files: list[Path]) -> dict[str, object]:
    structures = [_load(file) for file in files]
    scores = np.eye(len(files), dtype=np.float64)
    pairwise = []
    for left in range(len(files)):
        for right in range(left + 1, len(files)):
            score = _pair_score(structures[left], structures[right])
            scores[left, right] = scores[right, left] = score
            pairwise.append(score)
    thresholds = {}
    for threshold in (0.5, 0.6, 0.7):
        groups = _clusters(scores, threshold)
        thresholds[str(threshold)] = {
            "cluster_count": len(groups),
            "effective_cluster_count": _effective_cluster_count(groups),
            "largest_cluster_size": len(groups[0]),
            "largest_cluster_fraction": len(groups[0]) / len(files),
            "cluster_sizes": [len(group) for group in groups],
            "clusters": [[files[index].stem for index in group] for group in groups],
        }
    histogram = Counter(round(score, 1) for score in pairwise)
    return {
        "sample_count": len(files),
        "pair_count": len(pairwise),
        "tm_mean": statistics.fmean(pairwise),
        "tm_median": statistics.median(pairwise),
        "tm_min": min(pairwise),
        "tm_max": max(pairwise),
        "tm_histogram_0p1": {str(key): value for key, value in sorted(histogram.items())},
        "thresholds": thresholds,
        "score_matrix": scores.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--esmfold-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generated = sorted(args.sample_dir.glob("sample_*.pdb"))
    refolded = sorted((args.esmfold_dir / "predictions").glob("sample_*.pdb"))
    if len(generated) != args.expected_count or len(refolded) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} generated/refolded structures, "
            f"found {len(generated)}/{len(refolded)}"
        )
    if [file.stem for file in generated] != [file.stem for file in refolded]:
        raise ValueError("generated and refolded sample identities differ")
    result = {
        "method": "Biotite superimpose_structural_homologs, TM-align-inspired 3Di alignment",
        "generated": analyze(generated),
        "esmfold": analyze(refolded),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
