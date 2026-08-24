#!/usr/bin/env python3
"""Correct ESMFold summary pLDDT using only emitted protein atoms.

ESMFold's ``plddt`` output has Atom37 shape.  Its canonical mean excludes atom
slots that do not exist for a residue.  ``output_to_pdb`` applies that same
Atom37 existence mask, so averaging the B factors of the emitted ``ATOM``
records is equivalent, apart from PDB's two-decimal rounding.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


def mean_atom_plddt(pdb_file: Path) -> float:
    """Return mean pLDDT over real protein atoms in an ESMFold PDB."""

    values: list[float] = []
    for line in pdb_file.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  "):
            continue
        if line[16:17] not in {"", " ", "A"}:
            continue
        try:
            value = float(line[60:66])
        except ValueError as error:
            raise ValueError(f"invalid ATOM B factor in {pdb_file}: {line!r}") from error
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"invalid ATOM pLDDT in {pdb_file}: {value!r}")
        values.append(value)
    if not values:
        raise ValueError(f"no protein ATOM pLDDT values in {pdb_file}")
    return math.fsum(values) / len(values)


def _pdb_file(summary_file: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    sibling = summary_file.parent / candidate.name
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(f"missing ESMFold PDB referenced by {summary_file}: {value}")


def correct_summary(summary_file: Path) -> int:
    """Atomically replace unmasked means in an ESMFold ``summary.csv``."""

    with summary_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    required = {"id", "pdb", "mean_plddt"}
    if not required.issubset(fieldnames):
        missing = sorted(required - set(fieldnames))
        raise ValueError(f"{summary_file} lacks required columns: {missing}")
    if "mean_plddt_unmasked" not in fieldnames:
        fieldnames.append("mean_plddt_unmasked")

    for row in rows:
        row.setdefault("mean_plddt_unmasked", row["mean_plddt"])
        score = mean_atom_plddt(_pdb_file(summary_file, row["pdb"]))
        row["mean_plddt"] = f"{score:.3f}"

    temporary = summary_file.with_suffix(summary_file.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, summary_file)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("esmfold_out"))
    args, _unknown = parser.parse_known_args()
    count = correct_summary(args.outdir / "summary.csv")
    print(f"Corrected masked mean pLDDT for {count} ESMFold prediction(s)")


if __name__ == "__main__":
    main()
