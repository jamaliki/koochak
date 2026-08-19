#!/usr/bin/env python3
"""Report offline patch-count distributions after configured data filters."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.data import index_shards
from hierarchical_kaveh.model.patch import bucket_patch_capacity


def _patch_count(chains: np.ndarray, residues: np.ndarray) -> int:
    if not len(residues):
        return 0
    starts = np.ones(len(residues), dtype=np.bool_)
    starts[1:] = (chains[1:] != chains[:-1]) | (residues[1:] != residues[:-1] + 1)
    segment_starts = np.flatnonzero(starts)
    segment_ends = np.concatenate((segment_starts[1:], [len(residues)]))
    return int(np.sum((segment_ends - segment_starts + 3) // 4))


def audit(config_file: Path) -> dict[str, object]:
    config = load_config(config_file)
    references = index_shards(
        config.data.metadata_path,
        min_length=config.data.min_length,
        max_length=config.data.max_length,
        mean_plddt_min=config.data.mean_plddt_min,
        loop_length_max=config.data.loop_length_max,
        loop_content_max=config.data.loop_content_max,
        packing_density_min=config.data.packing_density_min,
    )
    buckets = config.data.effective_length_buckets
    counts: dict[int, list[int]] = {edge: [] for edge in buckets}
    by_shard = defaultdict(list)
    for reference in references:
        by_shard[reference.shard].append(reference)
    discontinuous_samples = 0
    for shard, shard_references in by_shard.items():
        with np.load(shard, allow_pickle=False) as payload:
            mask = payload["mask"]
            chain_index = payload["chain_idx"]
            residue_index = payload["res_idx"]
            for reference in shard_references:
                resolved = mask[reference.start:reference.stop, 1].astype(
                    np.bool_, copy=False
                )
                count = _patch_count(
                    chain_index[reference.start:reference.stop][resolved],
                    residue_index[reference.start:reference.stop][resolved],
                )
                compact = (reference.resolved_length + 3) // 4
                discontinuous_samples += count > compact
                edge = next(
                    edge for edge in buckets if reference.resolved_length <= edge
                )
                counts[edge].append(count)

    distributions: dict[str, object] = {}
    capacities: list[int] = []
    for edge in buckets:
        values = np.asarray(counts[edge], dtype=np.int64)
        required = max(
            int(values.max()) if len(values) else 0,
            (edge + 3) // 4,
        )
        capacities.append(bucket_patch_capacity(required))
        distributions[str(edge)] = {
            "samples": int(len(values)),
            "max": int(values.max()) if len(values) else None,
            "p50": float(np.quantile(values, 0.50)) if len(values) else None,
            "p95": float(np.quantile(values, 0.95)) if len(values) else None,
            "p99": float(np.quantile(values, 0.99)) if len(values) else None,
            "p999": float(np.quantile(values, 0.999)) if len(values) else None,
        }
    return {
        "eligible_samples": len(references),
        "discontinuous_samples": discontinuous_samples,
        "length_buckets": list(buckets),
        "patch_capacities": capacities,
        "distributions": distributions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.config)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
