from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_esmfold_history import audit


def _pdb_line(serial: int, atom: str, plddt: float) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4} ALA A   1    "
        f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{plddt:6.2f}          C  "
    )


def test_audit_recomputes_thresholds_without_mutating_inputs(tmp_path: Path) -> None:
    panel = tmp_path / "campaign/folds/cell"
    predictions = panel / "predictions"
    predictions.mkdir(parents=True)
    pdb_file = predictions / "sample_00000.pdb"
    pdb_file.write_text(
        _pdb_line(1, "N", 91.0) + "\n" + _pdb_line(2, "CA", 89.0) + "\n"
    )
    summary_file = predictions / "summary.csv"
    original_summary = (
        f"id,length,mean_plddt,pdb\nsample_00000,1,42.000,{pdb_file}\n"
    )
    summary_file.write_text(original_summary)
    (panel / "per_sample.json").write_text(json.dumps([{
        "id": "sample_00000",
        "mean_plddt": 42.0,
        "ca_rmsd_a": 1.0,
        "designable": 0,
        "sequence_sha256": "abc",
    }]))

    document, rows = audit(tmp_path, workers=1)

    assert summary_file.read_text() == original_summary
    assert document["prediction_count"] == 1
    assert document["campaigns"]["campaign"]["old_designable_count"] == 0
    assert document["campaigns"]["campaign"]["new_designable_count"] == 1
    assert rows[0]["corrected_mean_plddt"] == 90.0
