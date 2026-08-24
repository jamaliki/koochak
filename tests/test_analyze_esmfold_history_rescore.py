from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analyze_esmfold_history_rescore import analyze


def _row(panel: str, identifier: str, *, rmsd: float, old: float, corrected: float) -> dict:
    rmsd_pass = int(rmsd < 2)
    return {
        "campaign": "delayed-sidechain-offset-length128-esmfold",
        "panel": panel,
        "id": identifier,
        "original_mean_plddt": old,
        "corrected_mean_plddt": corrected,
        "ca_rmsd_a": rmsd,
        "rmsd_lt_2": rmsd_pass,
        "old_designable": int(rmsd_pass and old > 80),
        "new_designable": int(rmsd_pass and corrected > 80),
    }


def test_analyze_recomputes_ranking_and_deduplicates_panels(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    panels = []
    rows = []
    for arm, corrected, signature in (
        ("ratio1_control", 75.0, "control"),
        ("candidate", 90.0, "candidate"),
    ):
        panel = (
            "delayed-sidechain-offset-length128-esmfold/run/shards/"
            f"step200000/{arm}/L0064/predictions/summary.csv"
        )
        panels.append(
            {
                "campaign": "delayed-sidechain-offset-length128-esmfold",
                "panel": panel,
                "status": "ok",
                "matched_sample_count": 1,
                "panel_signature": signature,
                "canary": False,
            }
        )
        rows.append(_row(panel, "sample_00000", rmsd=1.0, old=70.0, corrected=corrected))
    audit = {
        "root": "/remote/root",
        "summary_file_count": 2,
        "esmfold_panel_count": 2,
        "error_panel_count": 0,
        "prediction_count": 2,
        "duplicate_panel_groups": [],
        "panels": panels,
    }
    (bundle / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (bundle / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = analyze([bundle], tmp_path / "unused-metadata")
    assert result["length128_offset"]["final_ranking"] == ["candidate", "ratio1_control"]
    candidate = result["length128_offset"]["steps"]["200000"]["candidate"]["aggregate"]
    assert candidate["old_designable_count"] == 0
    assert candidate["new_designable_count"] == 1


def test_analyze_groups_proteinmpnn_best_of_four(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    panel = "step-scale-mpnn-saveability/run/folds/scale2/predictions/summary.csv"
    panel_rows = []
    for sample_index in range(128):
        source_index = sample_index // 4
        success = source_index < 30 and sample_index % 4 == 0
        panel_rows.append(
            {
                **_row(panel, f"sample_{sample_index:05d}", rmsd=1.0, old=70, corrected=90 if success else 70),
                "campaign": "step-scale-mpnn-saveability",
            }
        )
    source_commit = "a" * 40
    source_panel = (
        "current-sampler-step-scale-designability/"
        f"{source_commit}/run/v1/folds/recycling_batch256_legacy/scale2/predictions/summary.csv"
    )
    source_rows = [
        {
            **_row(source_panel, f"sample_{sample_index:05d}", rmsd=1.0, old=70, corrected=70),
            "campaign": "current-sampler-step-scale-designability",
        }
        for sample_index in range(32)
    ]
    audit = {
        "root": "/remote/root",
        "summary_file_count": 2,
        "esmfold_panel_count": 2,
        "error_panel_count": 0,
        "prediction_count": 160,
        "duplicate_panel_groups": [],
        "panels": [{
            "campaign": "step-scale-mpnn-saveability",
            "panel": panel,
            "status": "ok",
            "matched_sample_count": 128,
            "panel_signature": "mpnn",
            "canary": False,
        }, {
            "campaign": "current-sampler-step-scale-designability",
            "panel": source_panel,
            "status": "ok",
            "matched_sample_count": 32,
            "panel_signature": "source",
            "canary": False,
        }],
    }
    (bundle / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (bundle / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in panel_rows + source_rows), encoding="utf-8"
    )
    metadata_dir = tmp_path / "metadata" / "scale2"
    metadata_dir.mkdir(parents=True)
    with (metadata_dir / "redesign_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_index", "source_sample_index", "source_pdb")
        )
        writer.writeheader()
        for sample_index in range(128):
            writer.writerow({
                "sample_index": sample_index,
                "source_sample_index": sample_index // 4,
                "source_pdb": (
                    "/remote/current-sampler-step-scale-designability/"
                    f"{source_commit}/run/v1/samples/recycling_batch256_legacy/scale2/"
                    f"sample_{sample_index // 4:05d}.pdb"
                ),
            })
    result = analyze([bundle], tmp_path / "metadata")
    assert result["proteinmpnn"]["scale2"]["per_sequence"]["new_designable_count"] == 30
    assert result["proteinmpnn"]["scale2"]["best_of_4"]["designable_backbone_count"] == 30
