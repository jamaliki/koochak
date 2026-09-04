#!/usr/bin/env python3
"""Score a paired step-scale ESMFold panel and its structural clusters."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_esmfold_designability_shard import (  # noqa: E402
    _pdb_ca_and_sequence,
    residue_indexed_tm_score,
)


RMSD_CUTOFF = 2.0
PLDDT_CUTOFF = 80.0
TM_CUTOFF = 0.5


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("at least one unique variant is required")
    return variants


def _load_rows(directory: Path, expected_count: int) -> list[dict[str, str]]:
    file = directory / "per_sample.csv"
    if not file.is_file():
        raise FileNotFoundError(file)
    with file.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["sample_index"]))
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows in {file}, found {len(rows)}")
    for row in rows:
        if row["status"] != "ok":
            raise ValueError(f"non-ok row in {file}: {row['id']}")
    return rows


def _clusters(scores: list[list[float]]) -> list[list[int]]:
    adjacency = {index: set() for index in range(len(scores))}
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            if scores[left][right] > TM_CUTOFF:
                adjacency[left].add(right)
                adjacency[right].add(left)
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in range(len(scores)):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        group = []
        while queue:
            current = queue.popleft()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: (-len(group), group))


def _score_variant(directory: Path, expected_count: int) -> dict[str, object]:
    rows = _load_rows(directory, expected_count)
    foldable = [
        row
        for row in rows
        if float(row["ca_rmsd_a"]) < RMSD_CUTOFF
        and float(row["mean_plddt"]) > PLDDT_CUTOFF
    ]
    structures = []
    for row in rows:
        pdb = directory / "predictions" / f"sample_{int(row['sample_index']):05d}.pdb"
        coordinates, sequence = _pdb_ca_and_sequence(pdb)
        structures.append((coordinates, sequence))
    lengths = {len(coordinates) for coordinates, _sequence in structures}
    if lengths != {len(structures[0][0])}:
        raise ValueError(f"inconsistent CA lengths in {directory}: {sorted(lengths)}")
    scores = [[1.0 if left == right else 0.0 for right in range(expected_count)] for left in range(expected_count)]
    pairwise = []
    for left in range(expected_count):
        for right in range(left + 1, expected_count):
            score = residue_indexed_tm_score(structures[left][0], structures[right][0])
            scores[left][right] = score
            scores[right][left] = score
            pairwise.append(score)
    clusters = _clusters(scores)
    return {
        "sample_count": len(rows),
        "foldable_count": len(foldable),
        "foldability_rate": len(foldable) / len(rows),
        "mean_ca_rmsd_a": statistics.fmean(float(row["ca_rmsd_a"]) for row in rows),
        "mean_plddt": statistics.fmean(float(row["mean_plddt"]) for row in rows),
        "pairwise_tm_count": len(pairwise),
        "pairwise_tm_mean": statistics.fmean(pairwise),
        "pairwise_tm_median": statistics.median(pairwise),
        "pairwise_tm_min": min(pairwise),
        "pairwise_tm_max": max(pairwise),
        "tm_edges_gt_0_5": sum(score > TM_CUTOFF for score in pairwise),
        "cluster_count": len(clusters),
        "cluster_sizes": [len(group) for group in clusters],
        "clusters": [[rows[index]["id"] for index in group] for group in clusters],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esmfold-root", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--step-scale", type=float, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = _parse_variants(args.variants)
    result = {
        "step": args.step,
        "step_scale": args.step_scale,
        "length": 128,
        "sample_count_per_variant": args.expected_count,
        "foldability_definition": "CA Kabsch RMSD < 2 A and mean ESMFold pLDDT > 80",
        "tm_definition": "campaign residue-indexed TM-like score after Kabsch alignment",
        "cluster_definition": "connected components of pairwise TM > 0.5 graph",
        "variants": {
            variant: _score_variant(args.esmfold_root / variant / "L0128", args.expected_count)
            for variant in variants
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
