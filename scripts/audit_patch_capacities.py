#!/usr/bin/env python3
"""Report offline patch-count distributions after configured data filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.data import index_shards
from hierarchical_kaveh.model.patch import bucket_patch_capacity


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
    for reference in references:
        edge = next(edge for edge in buckets if reference.resolved_length <= edge)
        counts[edge].append(reference.patch_count)

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
