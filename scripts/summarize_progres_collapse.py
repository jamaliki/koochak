#!/usr/bin/env python3
"""Aggregate checkpoint-wise Progres diversity, designability, and training losses."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


LOG_METRICS = (
    "loss",
    "coordinate_loss",
    "distogram_loss",
    "smooth_lddt_loss",
    "aatype_loss",
    "aatype_active_fraction",
    "step_time_s",
)


def _validate_ready(file: Path) -> dict[str, object]:
    manifest_file = file.with_name(file.name + ".ready.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    content = file.read_bytes()
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if Path(manifest["path"]) != file:
        raise ValueError(f"ready manifest path mismatch for {file}")
    if manifest["kind"] != "file" or manifest["counts"] != {"expected": 1, "observed": 1}:
        raise ValueError(f"ready manifest contract mismatch for {file}")
    if manifest["size_bytes"] != len(content) or manifest["sha256"] != observed_sha256:
        raise ValueError(f"ready manifest digest mismatch for {file}")
    return {
        "manifest": str(manifest_file),
        "sha256": observed_sha256,
        "size_bytes": len(content),
        "producer_workflow": manifest["provenance"]["workflow_id"],
        "producer_task": manifest["provenance"]["task_id"],
    }


def _mean_log_windows(log_file: Path, checkpoints: list[int], width: int) -> dict[int, dict[str, float]]:
    totals = {checkpoint: Counter() for checkpoint in checkpoints}
    counts = Counter()
    with log_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            for checkpoint in checkpoints:
                if checkpoint - width < step <= checkpoint:
                    counts[checkpoint] += 1
                    for metric in LOG_METRICS:
                        totals[checkpoint][metric] += float(row[metric])
    return {
        checkpoint: {
            metric: totals[checkpoint][metric] / counts[checkpoint]
            for metric in LOG_METRICS
        } | {"record_count": counts[checkpoint], "window_steps": width}
        for checkpoint in checkpoints
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def _cath_topology(notes: str) -> str:
    classification = notes.split(" - ", 1)[0].strip()
    return ".".join(classification.split(".")[:3])


def _panel_row(file: Path) -> dict[str, object]:
    result = json.loads(file.read_text(encoding="utf-8"))
    step = int(file.parent.parent.name.removeprefix("step"))
    cell = file.parent.name
    designable = set(result["designable_samples"])
    topology_counts = Counter(
        _cath_topology(hit["notes"])
        for hit in result["esmfold_cath40_nearest_hits"]
        if hit["sample"] in designable
    )
    top_topology, top_count = topology_counts.most_common(1)[0] if topology_counts else (None, 0)
    generated = result["generated"]["all_samples"]
    folded = result["esmfold"]["designable_subset"]
    sequence = result["generated_sequence_diversity"]
    return {
        "cell": cell,
        "step": step,
        "designable_count": result["designable_count"],
        "designability_fraction": result["designable_count"] / generated["sample_count"],
        "generated_progres_cluster_count": generated["cluster_count"],
        "generated_progres_effective_cluster_count": generated["effective_cluster_count"],
        "generated_progres_largest_cluster_fraction": generated["largest_cluster_fraction"],
        "generated_progres_same_fold_pair_fraction": generated["same_fold_pair_fraction"],
        "generated_progres_nearest_neighbor_similarity_mean": generated["nearest_neighbor_similarity_mean"],
        "designable_progres_cluster_count": folded["cluster_count"],
        "designable_progres_effective_cluster_count": folded["effective_cluster_count"],
        "designable_progres_largest_cluster_fraction": folded["largest_cluster_fraction"],
        "designable_progres_same_fold_pair_fraction": folded["same_fold_pair_fraction"],
        "sequence_unique_count": sequence["unique_sequence_count"],
        "sequence_mean_positional_entropy_bits": sequence["mean_positional_entropy_bits"],
        "sequence_pairwise_identity_mean": sequence["pairwise_sequence_identity_mean"],
        "designable_top_cath40_topology": top_topology,
        "designable_top_cath40_topology_count": top_count,
        "designable_top_cath40_topology_fraction": top_count / len(designable) if designable else None,
        "source": str(file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-data-progres", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--loss-window-steps", type=int, default=5_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.analysis_root.glob("step*/*/progres_diversity.json"))
    if len(files) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} panel results, found {len(files)}")
    ready_evidence = {str(file): _validate_ready(file) for file in files}
    training_data_ready = _validate_ready(args.training_data_progres)
    rows = [_panel_row(file) for file in files]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cell"])].append(row)
    correlations = {}
    for cell, cell_rows in grouped.items():
        cell_rows.sort(key=lambda row: int(row["step"]))
        checkpoints = [int(row["step"]) for row in cell_rows]
        windows = _mean_log_windows(args.training_root / cell / "log.csv", checkpoints, args.loss_window_steps)
        for row in cell_rows:
            row["training"] = windows[int(row["step"])]
        correlations[cell] = {
            "step_vs_designability": _pearson(checkpoints, [float(row["designability_fraction"]) for row in cell_rows]),
            "step_vs_generated_largest_cluster_fraction": _pearson(checkpoints, [float(row["generated_progres_largest_cluster_fraction"]) for row in cell_rows]),
            "step_vs_generated_effective_clusters": _pearson(checkpoints, [float(row["generated_progres_effective_cluster_count"]) for row in cell_rows]),
            "training_loss_vs_generated_largest_cluster_fraction": _pearson(
                [float(row["training"]["loss"]) for row in cell_rows],
                [float(row["generated_progres_largest_cluster_fraction"]) for row in cell_rows],
            ),
        }
    result = {
        "method": "Progres v1.1.0 checkpoint collapse history",
        "panel_count": len(rows),
        "loss_window_steps": args.loss_window_steps,
        "panels": sorted(rows, key=lambda row: (str(row["cell"]), int(row["step"]))),
        "correlations": correlations,
        "training_data": json.loads(args.training_data_progres.read_text(encoding="utf-8")),
        "input_ready_evidence": {
            "panels": ready_evidence,
            "training_data": training_data_ready,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat_rows = []
    for row in result["panels"]:
        training = row.pop("training")
        flat_rows.append(row | {f"train_{key}": value for key, value in training.items()})
        row["training"] = training
    table = args.output.with_suffix(".csv")
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"output": str(args.output), "table": str(table), "panel_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
