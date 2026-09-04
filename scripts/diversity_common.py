"""Dependency-light helpers shared by structural-diversity analyses."""

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import numpy as np


def clusters(scores: np.ndarray, threshold: float) -> list[list[int]]:
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


def effective_cluster_count(groups: list[list[int]]) -> float:
    total = sum(map(len, groups))
    probabilities = [len(group) / total for group in groups]
    return math.exp(-sum(probability * math.log(probability) for probability in probabilities))


def load_designable_stems(file: Path, expected_count: int) -> set[str]:
    with file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows in {file}, found {len(rows)}")
    return {
        f"sample_{int(row['sample_index']):05d}"
        for row in rows
        if row["status"] == "ok"
        and float(row["ca_rmsd_a"]) < 2.0
        and float(row["mean_plddt"]) > 80.0
    }
