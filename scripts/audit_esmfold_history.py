#!/usr/bin/env python3
"""Audit historical ESMFold panels with canonical Atom37-masked pLDDT."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.esmfold_plddt import mean_atom_plddt


def _prediction_file(summary_file: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    sibling = summary_file.parent / candidate.name
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(value)


def _sample_rows(summary_file: Path) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    candidates = (
        summary_file.parent.parent / "per_sample.json",
        summary_file.parent / "per_sample.json",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        document = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(document, list):
            continue
        rows = {str(row["id"]): row for row in document if isinstance(row, dict) and "id" in row}
        if len(rows) != len(document):
            raise ValueError(f"invalid or duplicate sample identifiers in {candidate}")
        return candidate, rows
    return None, {}


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def audit_summary(root: Path, summary_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    relative = summary_file.relative_to(root)
    with summary_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        summary_rows = list(reader)
    required = {"id", "pdb", "mean_plddt"}
    if not required.issubset(fieldnames):
        return {
            "panel": str(relative),
            "status": "not_esmfold",
            "columns": sorted(fieldnames),
        }, []
    if len({row["id"] for row in summary_rows}) != len(summary_rows):
        raise ValueError(f"duplicate ESMFold identifiers in {summary_file}")

    sample_file, samples = _sample_rows(summary_file)
    rows: list[dict[str, Any]] = []
    for summary_row in summary_rows:
        identifier = str(summary_row["id"])
        pdb_file = _prediction_file(summary_file, summary_row["pdb"])
        current_score = float(summary_row["mean_plddt"])
        original_score = float(summary_row.get("mean_plddt_unmasked") or current_score)
        corrected_score = mean_atom_plddt(pdb_file)
        sample = samples.get(identifier)
        rmsd = None if sample is None or "ca_rmsd_a" not in sample else float(sample["ca_rmsd_a"])
        rmsd_pass = None if rmsd is None else int(rmsd < 2.0)
        old_designable = None if rmsd_pass is None else int(bool(rmsd_pass) and original_score > 80.0)
        new_designable = None if rmsd_pass is None else int(bool(rmsd_pass) and corrected_score > 80.0)
        rows.append(
            {
                "campaign": relative.parts[0],
                "panel": str(relative),
                "id": identifier,
                "pdb": str(pdb_file),
                "original_mean_plddt": original_score,
                "current_mean_plddt": current_score,
                "corrected_mean_plddt": corrected_score,
                "already_corrected": abs(current_score - corrected_score) <= 0.02,
                "ca_rmsd_a": rmsd,
                "rmsd_lt_2": rmsd_pass,
                "old_designable": old_designable,
                "new_designable": new_designable,
                "stored_designable": None if sample is None else sample.get("designable"),
                "sequence_sha256": None if sample is None else sample.get("sequence_sha256"),
            }
        )

    decision_rows = [row for row in rows if row["ca_rmsd_a"] is not None]
    signature_values = [
        (
            row["id"],
            row["sequence_sha256"],
            round(float(row["ca_rmsd_a"]), 8) if row["ca_rmsd_a"] is not None else None,
        )
        for row in rows
    ]
    signature = hashlib.sha256(
        json.dumps(signature_values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    panel = {
        "campaign": relative.parts[0],
        "panel": str(relative),
        "status": "ok",
        "summary_file": str(summary_file),
        "sample_file": None if sample_file is None else str(sample_file),
        "prediction_count": len(rows),
        "matched_sample_count": len(decision_rows),
        "panel_signature": signature,
        "original_mean_plddt": _mean([float(row["original_mean_plddt"]) for row in rows]),
        "corrected_mean_plddt": _mean([float(row["corrected_mean_plddt"]) for row in rows]),
        "original_plddt_gt_80_count": sum(float(row["original_mean_plddt"]) > 80.0 for row in rows),
        "corrected_plddt_gt_80_count": sum(float(row["corrected_mean_plddt"]) > 80.0 for row in rows),
        "rmsd_lt_2_count": sum(int(row["rmsd_lt_2"]) for row in decision_rows),
        "old_designable_count": sum(int(row["old_designable"]) for row in decision_rows),
        "new_designable_count": sum(int(row["new_designable"]) for row in decision_rows),
        "already_corrected_count": sum(bool(row["already_corrected"]) for row in rows),
        "canary": "canary" in relative.parts,
    }
    return panel, rows


def _campaigns(panels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for campaign in sorted({str(panel["campaign"]) for panel in panels if "campaign" in panel}):
        selected = [
            panel for panel in panels
            if panel.get("campaign") == campaign and panel.get("status") == "ok" and not panel["canary"]
        ]
        result[campaign] = {
            "panel_count": len(selected),
            "prediction_count": sum(int(panel["prediction_count"]) for panel in selected),
            "matched_sample_count": sum(int(panel["matched_sample_count"]) for panel in selected),
            "old_designable_count": sum(int(panel["old_designable_count"]) for panel in selected),
            "new_designable_count": sum(int(panel["new_designable_count"]) for panel in selected),
        }
    return result


def audit(root: Path, *, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_files = sorted(root.rglob("summary.csv"))
    panels: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def inspect(summary_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            return audit_summary(root, summary_file)
        except Exception as error:  # keep the inventory complete and report failures
            return {
                "panel": str(summary_file.relative_to(root)),
                "campaign": summary_file.relative_to(root).parts[0],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }, []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for panel, panel_rows in executor.map(inspect, summary_files):
            panels.append(panel)
            rows.extend(panel_rows)
    panels.sort(key=lambda panel: str(panel["panel"]))
    rows.sort(key=lambda row: (str(row["panel"]), str(row["id"])))
    signatures: dict[str, list[str]] = {}
    for panel in panels:
        if panel.get("status") == "ok":
            signatures.setdefault(str(panel["panel_signature"]), []).append(str(panel["panel"]))
    duplicates = [group for group in signatures.values() if len(group) > 1]
    document = {
        "v": 1,
        "root": str(root),
        "summary_file_count": len(summary_files),
        "esmfold_panel_count": sum(panel.get("status") == "ok" for panel in panels),
        "error_panel_count": sum(panel.get("status") == "error" for panel in panels),
        "prediction_count": len(rows),
        "campaigns": _campaigns(panels),
        "duplicate_panel_groups": duplicates,
        "panels": panels,
    }
    return document, rows


def _markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Historical ESMFold pLDDT audit",
        "",
        f"Inventoried {document['summary_file_count']} summary files, "
        f"{document['esmfold_panel_count']} ESMFold panels, and "
        f"{document['prediction_count']} predictions.",
        "",
        "| Campaign | Panels | Predictions | Matched RMSD | Old strict | Corrected strict |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for campaign, metrics in document["campaigns"].items():
        lines.append(
            f"| `{campaign}` | {metrics['panel_count']} | {metrics['prediction_count']} | "
            f"{metrics['matched_sample_count']} | {metrics['old_designable_count']} | "
            f"{metrics['new_designable_count']} |"
        )
    lines.extend(("", f"Panel errors: {document['error_panel_count']}.", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    document, rows = audit(args.root, workers=args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(_markdown(document), encoding="utf-8")
    print(json.dumps({key: document[key] for key in (
        "summary_file_count", "esmfold_panel_count", "error_panel_count", "prediction_count"
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
