#!/usr/bin/env python3
"""Validate and aggregate the 16-cell Atom14 causal intervention endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any


SCHEMA = "hierarchical-kaveh.atom14-causal-intervention-aggregate.v1"
ANALYSIS_METHOD = "Progres protein graph embeddings"
EXPECTED_VERSION = "1.1.0"
EXPECTED_THRESHOLD = 0.8
EXPECTED_SAMPLE_COUNT = 32
EXPECTED_STEP = 50_000
PRIMARY_METRIC = "generated.designable_subset.complete_linkage_progres_cluster_count"
SECONDARY_METRIC = "esmfold.designable_subset.complete_linkage_progres_cluster_count"
CELL_PATTERN = re.compile(r"^(?P<architecture>.+)-o(?P<objective>[01])r(?P<residual>[01])t(?P<transport>[01])$")
ARCHITECTURES = {
    "flat_after_node_no_transition",
    "pool_before_attention_pair_transition",
}


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{label} is outside [{minimum}, {maximum}]")
    return value


def _at(document: dict[str, Any], dotted: str) -> Any:
    current: Any = document
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"analysis is missing {dotted}")
        current = current[part]
    return current


def _cell_factors(cell_id: str) -> dict[str, Any]:
    match = CELL_PATTERN.fullmatch(cell_id)
    if match is None or match.group("architecture") not in ARCHITECTURES:
        raise ValueError(f"unexpected cell ID: {cell_id}")
    return {
        "cell_id": cell_id,
        "architecture": match.group("architecture"),
        "objective": int(match.group("objective")),
        "residual": int(match.group("residual")),
        "transport": int(match.group("transport")),
    }


def _read_analysis(cell_id: str, file: Path, *, step: int) -> dict[str, Any]:
    document = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"analysis is not an object: {file}")
    _finite_tree(document, str(file))
    if document.get("method") != ANALYSIS_METHOD:
        raise ValueError(f"unexpected analysis method in {file}")
    if document.get("progres_version") != EXPECTED_VERSION:
        raise ValueError(f"unexpected Progres version in {file}")
    if document.get("same_fold_threshold") != EXPECTED_THRESHOLD:
        raise ValueError(f"unexpected Progres threshold in {file}")
    if document.get("primary_metric", {}).get("name") != PRIMARY_METRIC:
        raise ValueError(f"unexpected primary endpoint in {file}")
    if document.get("secondary_concordance_metric", {}).get("name") != SECONDARY_METRIC:
        raise ValueError(f"unexpected secondary endpoint in {file}")
    if document.get("panel") != cell_id or document.get("step") != step:
        raise ValueError(f"analysis identity mismatch in {file}")
    generated_count = _integer(
        _at(document, "generated.all_samples.sample_count"),
        f"{file}: generated sample count",
        maximum=EXPECTED_SAMPLE_COUNT,
    )
    refolded_count = _integer(
        _at(document, "esmfold.all_samples.sample_count"),
        f"{file}: ESMFold sample count",
        maximum=EXPECTED_SAMPLE_COUNT,
    )
    if generated_count != EXPECTED_SAMPLE_COUNT or refolded_count != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"analysis sample count is not {EXPECTED_SAMPLE_COUNT}: {file}")
    designable_count = _integer(
        document.get("designable_count"), f"{file}: designable_count", maximum=EXPECTED_SAMPLE_COUNT
    )
    generated_designable_count = _integer(
        _at(document, "generated.designable_subset.sample_count"),
        f"{file}: generated designable subset count",
        maximum=EXPECTED_SAMPLE_COUNT,
    )
    esmfold_designable_count = _integer(
        _at(document, "esmfold.designable_subset.sample_count"),
        f"{file}: ESMFold designable subset count",
        maximum=EXPECTED_SAMPLE_COUNT,
    )
    if {generated_designable_count, esmfold_designable_count} != {designable_count}:
        raise ValueError(f"designable subset counts disagree in {file}")
    primary = _integer(
        document.get("primary_designable_progres_cluster_count"),
        f"{file}: primary cluster count",
        maximum=designable_count,
    )
    if _integer(_at(document, "primary_metric.value"), f"{file}: primary metric value", maximum=designable_count) != primary:
        raise ValueError(f"primary metric value disagrees in {file}")
    secondary = _integer(
        document.get("secondary_esmfold_designable_progres_cluster_count"),
        f"{file}: secondary cluster count",
        maximum=designable_count,
    )
    if _integer(_at(document, "secondary_concordance_metric.value"), f"{file}: secondary metric value", maximum=designable_count) != secondary:
        raise ValueError(f"secondary metric value disagrees in {file}")
    sequence_count = _integer(
        _at(document, "generated_sequence_diversity.sample_count"),
        f"{file}: sequence sample count",
        maximum=EXPECTED_SAMPLE_COUNT,
    )
    if sequence_count != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"sequence sample count is not {EXPECTED_SAMPLE_COUNT}: {file}")
    return {
        "cell": _cell_factors(cell_id),
        "analysis_path": str(file.resolve()),
        "analysis_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "analysis": document,
        "metrics": _metrics(document, primary, secondary, designable_count),
    }


def _metrics(document: dict[str, Any], primary: int, secondary: int, designable_count: int) -> dict[str, Any]:
    generated = _at(document, "generated.designable_subset")
    refolded = _at(document, "esmfold.designable_subset")
    sequence = _at(document, "generated_sequence_diversity")
    return {
        "primary_designable_progres_cluster_count": primary,
        "secondary_esmfold_designable_progres_cluster_count": secondary,
        "designable_count": designable_count,
        "designability_fraction": designable_count / EXPECTED_SAMPLE_COUNT,
        "generated_designable_cluster_count": _integer(generated["cluster_count"], "generated cluster count", maximum=EXPECTED_SAMPLE_COUNT),
        "generated_designable_effective_cluster_count": _number(generated["effective_cluster_count"], "generated effective cluster count"),
        "generated_designable_largest_cluster_fraction": _optional_number(generated["largest_cluster_fraction"], "generated largest cluster fraction"),
        "generated_designable_same_fold_pair_fraction": _optional_number(generated["same_fold_pair_fraction"], "generated same-fold pair fraction"),
        "generated_designable_same_cluster_pair_fraction": _optional_number(generated["same_cluster_pair_fraction"], "generated same-cluster pair fraction"),
        "esmfold_designable_cluster_count": _integer(refolded["cluster_count"], "ESMFold cluster count", maximum=EXPECTED_SAMPLE_COUNT),
        "esmfold_designable_effective_cluster_count": _number(refolded["effective_cluster_count"], "ESMFold effective cluster count"),
        "esmfold_designable_largest_cluster_fraction": _optional_number(refolded["largest_cluster_fraction"], "ESMFold largest cluster fraction"),
        "generated_unique_sequence_count": _integer(sequence["unique_sequence_count"], "unique sequence count", maximum=EXPECTED_SAMPLE_COUNT),
        "generated_largest_identical_sequence_fraction": _number(sequence["largest_identical_sequence_fraction"], "largest identical sequence fraction") if sequence["largest_identical_sequence_fraction"] is not None else None,
        "generated_pairwise_sequence_identity_mean": _number(sequence["pairwise_sequence_identity_mean"], "pairwise sequence identity mean") if sequence["pairwise_sequence_identity_mean"] is not None else None,
    }


def _parse_inputs(values: list[str], *, expected_cells: int, step: int) -> list[dict[str, Any]]:
    if len(values) != expected_cells:
        raise ValueError(f"expected {expected_cells} --analysis inputs, found {len(values)}")
    seen: set[str] = set()
    rows = []
    for value in values:
        cell_id, separator, raw_file = value.partition("=")
        if not separator or not cell_id or not raw_file or cell_id in seen:
            raise ValueError("--analysis values must use unique CELL_ID=FILE entries")
        file = Path(raw_file).expanduser().resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        rows.append(_read_analysis(cell_id, file, step=step))
        seen.add(cell_id)
    expected_ids = {
        f"{architecture}-o{objective}r{residual}t{transport}"
        for architecture in sorted(ARCHITECTURES)
        for objective in (0, 1)
        for residual in (0, 1)
        for transport in (0, 1)
    }
    if seen != expected_ids:
        raise ValueError(f"analysis cells differ from the complete 16-cell factorial: {sorted(seen ^ expected_ids)}")
    return sorted(rows, key=lambda row: row["cell"]["cell_id"])


EFFECT_COLUMNS = (
    "cell_id", "architecture", "objective", "residual", "transport",
    "primary_designable_progres_cluster_count",
    "secondary_esmfold_designable_progres_cluster_count",
    "designable_count", "designability_fraction",
    "generated_designable_cluster_count",
    "generated_designable_effective_cluster_count",
    "generated_designable_largest_cluster_fraction",
    "generated_designable_same_fold_pair_fraction",
    "generated_designable_same_cluster_pair_fraction",
    "esmfold_designable_cluster_count",
    "esmfold_designable_effective_cluster_count",
    "esmfold_designable_largest_cluster_fraction",
    "generated_unique_sequence_count",
    "generated_largest_identical_sequence_fraction",
    "generated_pairwise_sequence_identity_mean",
)


def _effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row["cell"].get(key, row["metrics"].get(key)) for key in EFFECT_COLUMNS}
        for row in rows
    ]


def aggregate(values: list[str], *, output: Path, workflow: str, step: int = EXPECTED_STEP, expected_cells: int = 16) -> dict[str, Any]:
    rows = _parse_inputs(values, expected_cells=expected_cells, step=step)
    document = {
        "schema": SCHEMA,
        "workflow_id": workflow,
        "step": step,
        "expected_cell_count": expected_cells,
        "primary_endpoint": PRIMARY_METRIC,
        "clustering": {"method": "complete linkage", "same_fold_threshold": EXPECTED_THRESHOLD},
        "cell_analyses": rows,
        "effects_ready_columns": list(EFFECT_COLUMNS),
        "effects_ready": _effects(rows),
    }
    _finite_tree(document, "aggregate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", required=True, metavar="CELL_ID=FILE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--step", type=int, default=EXPECTED_STEP)
    parser.add_argument("--expected-cells", type=int, default=16)
    args = parser.parse_args(argv)
    document = aggregate(
        args.analysis,
        output=args.output,
        workflow=args.workflow,
        step=args.step,
        expected_cells=args.expected_cells,
    )
    print(json.dumps({"output": str(args.output.resolve()), "cell_count": len(document["cell_analyses"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
