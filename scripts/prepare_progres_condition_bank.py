#!/usr/bin/env python3
"""Validate the Progres database and select a reproducible diverse target bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from hierarchical_kaveh.data.progres import ProgresSidecarReader, sha256_file
from hierarchical_kaveh.data.shards import SampleReference, index_shards


SCHEMA = "progres-condition-bank-v1"
SELECTION_SEED = 20260905
TARGET_COUNT = 8


def _stable_key(reference: SampleReference) -> str:
    return f"{reference.shard.resolve()}::{reference.index}"


def _select_diverse(ids: list[str], vectors: np.ndarray, count: int, seed: int) -> list[int]:
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] != 128:
        raise ValueError("condition candidates must be an [M,128] matrix")
    if len(ids) < count:
        raise ValueError(f"condition bank needs at least {count} valid candidates")
    starts = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(f"{seed}:{ids[index]}".encode()).hexdigest(),
    )
    selected = [starts[0]]
    minimum_distance = 1.0 - vectors @ vectors[selected[0]]
    while len(selected) < count:
        candidates = [index for index in range(len(ids)) if index not in selected]
        chosen = max(candidates, key=lambda index: (float(minimum_distance[index]), ids[index]))
        selected.append(chosen)
        minimum_distance = np.minimum(minimum_distance, 1.0 - vectors @ vectors[chosen])
    return selected


def build_bank(index_path: Path, metadata_path: Path, output: Path, *, seed: int = SELECTION_SEED) -> dict[str, object]:
    reader = ProgresSidecarReader(index_path, metadata_path=metadata_path, eager=True)
    references = index_shards(
        metadata_path,
        min_length=128,
        max_length=128,
        mean_plddt_min=80.0,
        loop_length_max=None,
        loop_content_max=0.5,
        packing_density_min=None,
    )
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for reference in references:
        ids.append(reader.identifier(reference))
        vectors.append(reader.embedding(reference).numpy())
    matrix = np.asarray(vectors, dtype=np.float32)
    selected = _select_diverse(ids, matrix, TARGET_COUNT, seed)
    progres = reader.index["progres"]
    result = {
        "schema": SCHEMA,
        "selection": {
            "seed": seed,
            "count": TARGET_COUNT,
            "candidate_count": len(references),
            "filters": {
                "min_length": 128,
                "max_length": 128,
                "mean_plddt_min": 80.0,
                "loop_length_max": None,
                "loop_content_max": 0.5,
                "packing_density_min": None,
            },
            "method": "seeded_sha256_start_greedy_farthest_cosine",
        },
        "sidecar_index": str(index_path.resolve()),
        "sidecar_index_sha256": sha256_file(index_path),
        "metadata": reader.index["metadata"],
        "progres": progres,
        "targets": [
            {
                "target_id": ids[index],
                "rank": rank,
                "embedding": [float(value) for value in matrix[index]],
            }
            for rank, index in enumerate(selected)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "candidate_count": len(references), "target_count": len(selected)}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    args = parser.parse_args()
    build_bank(args.index, args.metadata, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
