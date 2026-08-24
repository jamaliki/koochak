from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.esmfold_plddt import correct_summary, mean_atom_plddt


def _pdb_line(
    serial: int,
    atom: str,
    *,
    plddt: float,
    record: str = "ATOM",
    altloc: str = " ",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:^4}{altloc}ALA A   1    "
        f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{plddt:6.2f}          C  "
    )


def test_mean_atom_plddt_uses_only_real_primary_atoms(tmp_path: Path) -> None:
    pdb_file = tmp_path / "prediction.pdb"
    pdb_file.write_text(
        "\n".join(
            (
                _pdb_line(1, "N", plddt=90.0),
                _pdb_line(2, "CA", plddt=80.0),
                _pdb_line(3, "CB", plddt=0.0, altloc="B"),
                _pdb_line(4, "C1", plddt=0.0, record="HETATM"),
            )
        )
        + "\n"
    )
    assert mean_atom_plddt(pdb_file) == pytest.approx(85.0)


def test_correct_summary_preserves_unmasked_value_and_is_idempotent(
    tmp_path: Path,
) -> None:
    pdb_file = tmp_path / "sample.pdb"
    pdb_file.write_text(
        _pdb_line(1, "N", plddt=91.0)
        + "\n"
        + _pdb_line(2, "CA", plddt=89.0)
        + "\n"
    )
    summary_file = tmp_path / "summary.csv"
    summary_file.write_text(
        f"id,length,mean_plddt,pdb\nsample,1,42.000,{pdb_file}\n"
    )

    assert correct_summary(summary_file) == 1
    assert correct_summary(summary_file) == 1
    with summary_file.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["mean_plddt"] == "90.000"
    assert row["mean_plddt_unmasked"] == "42.000"
