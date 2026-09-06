#!/usr/bin/env python3
"""Aggregate complete-linkage Progres reanalysis outputs and availability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _row(entry: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    generated = result["generated"]
    generated_all = generated["all_samples"]
    primary = generated["designable_subset"]
    secondary = result["esmfold"]["designable_subset"]
    return {
        "panel_id": entry["panel_id"],
        "category": entry["category"],
        "step": entry.get("step"),
        "architecture": entry.get("architecture"),
        "filter_regime": entry.get("filter_regime"),
        "self_conditioning_probability": entry.get("self_conditioning_probability"),
        "coverage_status": entry.get("coverage_status"),
        "availability_status": entry.get("availability_status"),
        "analysis_status": "available",
        "source_input": entry.get("source_input"),
        "analysis_output": entry.get("analysis_output"),
        "designable_count": result["designable_count"],
        "all_sample_count": generated_all["sample_count"],
        "all_sample_cluster_count": generated_all["cluster_count"],
        "all_sample_effective_cluster_count": generated_all["effective_cluster_count"],
        "all_sample_largest_cluster_fraction": generated_all["largest_cluster_fraction"],
        "primary_generated_designable_progres_cluster_count": primary["cluster_count"],
        "primary_generated_designable_cluster_sizes": primary["cluster_sizes"],
        "primary_generated_designable_effective_cluster_count": primary["effective_cluster_count"],
        "primary_generated_designable_largest_cluster_fraction": primary["largest_cluster_fraction"],
        "primary_generated_designable_medoids": primary.get("cluster_medoids"),
        "primary_generated_designable_same_cluster_pair_fraction": primary.get("same_cluster_pair_fraction"),
        "secondary_esmfold_designable_progres_cluster_count": secondary["cluster_count"],
        "secondary_esmfold_designable_cluster_sizes": secondary["cluster_sizes"],
        "secondary_esmfold_designable_effective_cluster_count": secondary["effective_cluster_count"],
        "secondary_esmfold_designable_largest_cluster_fraction": secondary["largest_cluster_fraction"],
        "secondary_esmfold_designable_medoids": secondary.get("cluster_medoids"),
        "secondary_esmfold_designable_same_cluster_pair_fraction": secondary.get("same_cluster_pair_fraction"),
        "designable_progres_cluster_count_disagreement": (
            primary["cluster_count"] != secondary["cluster_count"]
        ),
    }


def summarize(manifest_file: Path) -> dict[str, object]:
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "progres-complete-linkage-availability-v1":
        raise ValueError("availability manifest schema mismatch")
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for entry in manifest["entries"]:
        output = Path(str(entry["analysis_output"]))
        if output.is_file():
            result = json.loads(output.read_text(encoding="utf-8"))
            if result.get("primary_metric", {}).get("name") != (
                "generated.designable_subset.complete_linkage_progres_cluster_count"
            ):
                raise ValueError(f"primary endpoint missing from {output}")
            rows.append(_row(entry, result))
        else:
            missing.append(
                {
                    "panel_id": entry["panel_id"],
                    "coverage_status": entry.get("coverage_status"),
                    "availability_status": entry.get("availability_status"),
                    "missing_paths": entry.get("missing_paths", []),
                    "analysis_output": str(output),
                    "reason": (
                        "missing_input"
                        if entry.get("availability_status") == "missing"
                        else "missing_analysis_output"
                    ),
                }
            )
    disagreements = [
        row["panel_id"] for row in rows
        if row["designable_progres_cluster_count_disagreement"]
    ]
    result = {
        "schema": "progres-complete-linkage-summary-v1",
        "algorithm": "deterministic complete linkage on Progres similarity at >= 0.8; generated embeddings restricted by ESMFold designable IDs are primary",
        "primary_metric": "generated.designable_subset.complete_linkage_progres_cluster_count",
        "secondary_metric": "esmfold.designable_subset.complete_linkage_progres_cluster_count",
        "panel_count": len(manifest["entries"]),
        "unique_analysis_count": len({entry["analysis_task_id"] for entry in manifest["entries"] if entry.get("analysis_task_id")}),
        "available_panel_count": len(rows),
        "missing_or_failed_panel_count": len(missing),
        "disagreement_count": len(disagreements),
        "disagreement_panel_ids": disagreements,
        "panels": sorted(rows, key=lambda row: str(row["panel_id"])),
        "missing_or_failed": sorted(missing, key=lambda row: str(row["panel_id"])),
        "availability_manifest": str(manifest_file.resolve()),
        "availability": {
            "declared_entry_count": manifest["entry_count"],
            "declared_available_count": manifest["available_count"],
            "declared_missing_count": manifest["missing_count"],
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.availability_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = args.output.with_suffix(".csv")
    rows = result["panels"]
    if rows:
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "output": str(args.output),
        "table": str(table),
        "available_panel_count": result["available_panel_count"],
        "missing_or_failed_panel_count": result["missing_or_failed_panel_count"],
        "disagreement_count": result["disagreement_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
