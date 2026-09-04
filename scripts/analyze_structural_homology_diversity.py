#!/usr/bin/env python3
"""Cluster generated proteins with structural-alignment TM scores."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.tm import superimpose_structural_homologs, tm_score

from scripts.diversity_common import (
    clusters as _clusters,
)
from scripts.diversity_common import (
    effective_cluster_count as _effective_cluster_count,
)
from scripts.diversity_common import (
    load_designable_stems as _load_designable_stems,
)


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


def _summary(files: list[Path], scores: np.ndarray, indices: list[int]) -> dict[str, object]:
    subset = scores[np.ix_(indices, indices)]
    pairwise = [subset[left, right] for left in range(len(indices)) for right in range(left + 1, len(indices))]
    thresholds = {}
    for threshold in (0.5, 0.6, 0.7):
        groups = _clusters(subset, threshold)
        thresholds[str(threshold)] = {
            "cluster_count": len(groups),
            "effective_cluster_count": _effective_cluster_count(groups) if groups else 0.0,
            "largest_cluster_size": len(groups[0]) if groups else 0,
            "largest_cluster_fraction": len(groups[0]) / len(indices) if groups else None,
            "cluster_sizes": [len(group) for group in groups],
            "clusters": [[files[indices[index]].stem for index in group] for group in groups],
        }
    histogram = Counter(round(score, 1) for score in pairwise)
    return {
        "sample_count": len(indices),
        "pair_count": len(pairwise),
        "tm_mean": statistics.fmean(pairwise) if pairwise else None,
        "tm_median": statistics.median(pairwise) if pairwise else None,
        "tm_min": min(pairwise) if pairwise else None,
        "tm_max": max(pairwise) if pairwise else None,
        "tm_histogram_0p1": {str(key): value for key, value in sorted(histogram.items())},
        "thresholds": thresholds,
    }


def analyze(files: list[Path], designable_stems: set[str] | None = None) -> dict[str, object]:
    structures = [_load(file) for file in files]
    scores = np.eye(len(files), dtype=np.float64)
    alignment_failures = []
    for left in range(len(files)):
        for right in range(left + 1, len(files)):
            try:
                score = _pair_score(structures[left], structures[right])
            except ValueError as error:
                if "No anchors found" not in str(error):
                    raise
                score = 0.0
                alignment_failures.append([files[left].stem, files[right].stem])
            scores[left, right] = scores[right, left] = score
    result = {
        **_summary(files, scores, list(range(len(files)))),
        "no_anchor_pair_count": len(alignment_failures),
        "no_anchor_pairs": alignment_failures,
        "score_matrix": scores.tolist(),
    }
    if designable_stems is not None:
        indices = [index for index, file in enumerate(files) if file.stem in designable_stems]
        result["designable_subset"] = _summary(files, scores, indices)
    return result


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
    designable_stems = _load_designable_stems(
        args.esmfold_dir / "per_sample.csv", args.expected_count,
    )
    result = {
        "method": "Biotite superimpose_structural_homologs, TM-align-inspired 3Di alignment",
        "designability_definition": "CA RMSD < 2 A and mean ESMFold pLDDT > 80",
        "designable_count": len(designable_stems),
        "designable_samples": sorted(designable_stems),
        "generated": analyze(generated, designable_stems),
        "esmfold": analyze(refolded, designable_stems),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
