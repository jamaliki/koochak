#!/usr/bin/env python3
"""Annotate ragged-shard metadata with invalid CA-distance sample indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from hierarchical_kaveh.data.shards import (
    CA_DISTANCE_EXCLUSIONS_KEY,
    CA_DISTANCE_VALIDATION_KEY,
    MAX_CONSECUTIVE_CA_DISTANCE,
    invalid_ca_distance_samples,
)


def _entries(document: object) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(document, list):
        return document, None
    if isinstance(document, dict):
        for key in ("shards", "entries"):
            value = document.get(key)
            if isinstance(value, list):
                return value, key
    raise ValueError("metadata must be a list, or a mapping containing 'shards' or 'entries'")


def materialize_metadata(metadata_path: Path, output_path: Path) -> dict[str, int | float]:
    """Write a metadata copy carrying exact per-shard exclusion indices."""

    metadata_path = metadata_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path == metadata_path:
        raise ValueError("output must differ from the source metadata path")
    if output_path.parent != metadata_path.parent:
        raise ValueError("output must share the metadata directory so relative shard paths remain valid")
    document = json.loads(metadata_path.read_text())
    entries, container_key = _entries(document)
    annotated: list[dict[str, Any]] = []
    total_samples = 0
    total_excluded = 0
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "shard" not in entry:
            raise ValueError(f"metadata entry {entry_index} is invalid")
        shard = Path(str(entry["shard"]))
        if not shard.is_absolute():
            shard = metadata_path.parent / shard
        with np.load(shard, allow_pickle=False) as payload:
            offsets = np.asarray(payload["sample_offsets"], dtype=np.int64)
            excluded = invalid_ca_distance_samples(
                payload["pos"],
                payload["mask"],
                payload["chain_idx"],
                payload["res_idx"],
                offsets,
            )
        result = dict(entry)
        result[CA_DISTANCE_VALIDATION_KEY] = MAX_CONSECUTIVE_CA_DISTANCE
        result[CA_DISTANCE_EXCLUSIONS_KEY] = excluded.tolist()
        annotated.append(result)
        total_samples += len(offsets) - 1
        total_excluded += len(excluded)
        if (entry_index + 1) % 100 == 0:
            print(f"validated {entry_index + 1}/{len(entries)} shards", file=sys.stderr)

    if container_key is None:
        output_document: object = annotated
    else:
        output_document = {**document, container_key: annotated}  # type: ignore[arg-type]
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(json.dumps(output_document, separators=(",", ":")))
    temporary.replace(output_path)
    return {
        "shards": len(annotated),
        "samples": total_samples,
        "excluded_samples": total_excluded,
        "maximum_ca_distance": MAX_CONSECUTIVE_CA_DISTANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        parser.error(f"output already exists: {output}; pass --overwrite to replace it")
    summary = materialize_metadata(arguments.metadata, output)
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
