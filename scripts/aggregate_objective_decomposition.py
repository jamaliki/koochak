#!/usr/bin/env python3
"""Validate and aggregate the lDDT decomposition and C-alpha ablation endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any


SCHEMA = "hierarchical-kaveh.objective-decomposition-ca-ablation.v1"
DECOMPOSITION_PATTERN = re.compile(r"^lddt-m(?P<mask>[01])g(?P<gate>[01])c(?P<compensation>[01])$")
ABLATION_CELLS = {"coordseq-atom14", "coordseq-ca"}
EXPECTED_METHOD = "Progres protein graph embeddings"
EXPECTED_VERSION = "1.1.0"
EXPECTED_STEP = 50_000
EXPECTED_SAMPLES = 32
EXPECTED_THRESHOLD = 0.8


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")


def _at(document: dict[str, Any], dotted: str) -> Any:
    current: Any = document
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"analysis is missing {dotted}")
        current = current[part]
    return current


def _integer(value: Any, label: str, *, maximum: int = EXPECTED_SAMPLES) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [0, {maximum}]")
    return value


def _parse_specification(value: str) -> tuple[str, Path]:
    cell_id, separator, raw_file = value.partition("=")
    if not separator or not cell_id or not raw_file:
        raise ValueError("--analysis values must use CELL_ID=FILE")
    return cell_id, Path(raw_file).expanduser().resolve()


def _factors(cell_id: str) -> dict[str, Any]:
    match = DECOMPOSITION_PATTERN.fullmatch(cell_id)
    if match is not None:
        return {
            "family": "lddt_decomposition",
            "physical_atom_mask": int(match.group("mask")),
            "sigma_le_3_gate": int(match.group("gate")),
            "inverse_c_out_compensation": int(match.group("compensation")),
            "atom_representation": "atom14",
            "objectives": "coordinate+sequence+lddt+distogram",
        }
    if cell_id in ABLATION_CELLS:
        return {
            "family": "coordinate_sequence_ablation",
            "physical_atom_mask": None,
            "sigma_le_3_gate": None,
            "inverse_c_out_compensation": None,
            "atom_representation": cell_id.removeprefix("coordseq-"),
            "objectives": "coordinate+sequence",
        }
    raise ValueError(f"unexpected cell ID: {cell_id}")


def _read(cell_id: str, file: Path) -> dict[str, Any]:
    if not file.is_file():
        raise FileNotFoundError(file)
    document = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"analysis is not an object: {file}")
    _finite_tree(document, str(file))
    expected = {
        "method": EXPECTED_METHOD,
        "progres_version": EXPECTED_VERSION,
        "same_fold_threshold": EXPECTED_THRESHOLD,
        "step": EXPECTED_STEP,
    }
    for key, wanted in expected.items():
        if document.get(key) != wanted:
            raise ValueError(f"{file}: expected {key}={wanted!r}, got {document.get(key)!r}")
    for dotted in ("generated.all_samples.sample_count", "esmfold.all_samples.sample_count"):
        if _integer(_at(document, dotted), f"{file}:{dotted}") != EXPECTED_SAMPLES:
            raise ValueError(f"{file}: {dotted} is not {EXPECTED_SAMPLES}")
    designable = _integer(document.get("designable_count"), f"{file}:designable_count")
    clusters = _integer(
        document.get("primary_designable_progres_cluster_count"),
        f"{file}:primary_designable_progres_cluster_count",
        maximum=designable,
    )
    all_clusters = _integer(
        _at(document, "generated.all_samples.cluster_count"),
        f"{file}:generated.all_samples.cluster_count",
    )
    return {
        "cell_id": cell_id,
        "factors": _factors(cell_id),
        "source_panel": document.get("panel"),
        "analysis_path": str(file),
        "analysis_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "designable_count": designable,
        "designability_fraction": designable / EXPECTED_SAMPLES,
        "designable_progres_cluster_count": clusters,
        "all_sample_progres_cluster_count": all_clusters,
        "analysis": document,
    }


def aggregate(specifications: list[str], *, workflow: str, output: Path) -> dict[str, Any]:
    parsed = [_parse_specification(value) for value in specifications]
    if len({cell_id for cell_id, _ in parsed}) != len(parsed):
        raise ValueError("analysis cell IDs must be unique")
    expected = {
        *(f"lddt-m{mask}g{gate}c{compensation}" for mask in (0, 1) for gate in (0, 1) for compensation in (0, 1)),
        *ABLATION_CELLS,
    }
    observed = {cell_id for cell_id, _ in parsed}
    if observed != expected:
        raise ValueError(f"analysis cells differ from the ten-cell endpoint: {sorted(observed ^ expected)}")
    rows = sorted((_read(cell_id, file) for cell_id, file in parsed), key=lambda row: row["cell_id"])
    document = {
        "schema": SCHEMA,
        "workflow_id": workflow,
        "step": EXPECTED_STEP,
        "sample_count_per_cell": EXPECTED_SAMPLES,
        "primary_endpoint": "number of designable Progres clusters",
        "cells": rows,
    }
    _finite_tree(document, "aggregate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", required=True, metavar="CELL_ID=FILE")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = aggregate(args.analysis, workflow=args.workflow, output=args.output)
    print(json.dumps({"output": str(args.output.resolve()), "cell_count": len(document["cells"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
