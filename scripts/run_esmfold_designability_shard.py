#!/usr/bin/env python3
"""Refold one generated sample shard with ESMFold and score self-consistency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


RESTYPE_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
DESIGNABLE_RMSD_A = 2.0
DESIGNABLE_PLDDT = 80.0


def _atomic_write_text(file: Path, value: str) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, file)


def _atomic_write_json(file: Path, value: object) -> None:
    _atomic_write_text(file, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_fasta(file: Path) -> str:
    sequence = "".join(
        line.strip()
        for line in file.read_text().splitlines()
        if not line.startswith(">")
    ).replace("/", "")
    if not sequence or any(letter not in STANDARD_AA for letter in sequence):
        raise ValueError(f"{file} does not contain one standard protein sequence")
    return sequence


def _pdb_ca_and_sequence(file: Path) -> tuple[np.ndarray, str]:
    residues: list[tuple[tuple[str, str, str], str, tuple[float, float, float]]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in file.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[16:17] not in {"", " ", "A"}:
            continue
        residue_key = (line[21:22], line[22:26], line[26:27])
        if residue_key in seen:
            continue
        seen.add(residue_key)
        residue_name = line[17:20].strip()
        if residue_name not in RESTYPE_3_TO_1:
            raise ValueError(f"unsupported residue {residue_name!r} in {file}")
        residues.append(
            (
                residue_key,
                RESTYPE_3_TO_1[residue_name],
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            )
        )
    if len(residues) < 3:
        raise ValueError(f"need at least three CA atoms in {file}")
    coordinates = np.asarray([record[2] for record in residues], dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"non-finite CA coordinates in {file}")
    return coordinates, "".join(record[1] for record in residues)


def _kabsch_aligned(
    target: np.ndarray,
    mobile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if target.shape != mobile.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Kabsch inputs must have the same [N,3] shape")
    target_centered = target - target.mean(axis=0, keepdims=True)
    mobile_centered = mobile - mobile.mean(axis=0, keepdims=True)
    left, _singular_values, right = np.linalg.svd(mobile_centered.T @ target_centered)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return target_centered, mobile_centered @ rotation


def kabsch_rmsd(target: np.ndarray, mobile: np.ndarray) -> float:
    """Return residue-indexed CA RMSD after rigid Kabsch alignment."""

    target_aligned, mobile_aligned = _kabsch_aligned(target, mobile)
    difference = target_aligned - mobile_aligned
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=-1))))


def residue_indexed_tm_score(target: np.ndarray, mobile: np.ndarray) -> float:
    """Return a residue-indexed TM-like score after shared-sequence alignment."""

    target_aligned, mobile_aligned = _kabsch_aligned(target, mobile)
    length = int(target_aligned.shape[0])
    d0 = 0.5 if length <= 21 else max(0.5, 1.24 * ((length - 15) ** (1 / 3)) - 1.8)
    distances = np.sqrt(np.sum((target_aligned - mobile_aligned) ** 2, axis=-1))
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def _sample_records(
    sample_dir: Path,
    *,
    expected_count: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    fasta_files = sorted(sample_dir.glob("sample_*.fasta"))
    if len(fasta_files) != expected_count:
        raise ValueError(
            f"expected {expected_count} FASTA files in {sample_dir}, found {len(fasta_files)}"
        )
    if limit is not None:
        if limit <= 0 or limit > len(fasta_files):
            raise ValueError(
                "limit must be positive and no greater than expected-count"
            )
        fasta_files = fasta_files[:limit]
    records = []
    for fasta in fasta_files:
        pdb = fasta.with_suffix(".pdb")
        if not pdb.is_file():
            raise FileNotFoundError(f"missing paired PDB: {pdb}")
        sequence = _read_fasta(fasta)
        generated_ca, generated_sequence = _pdb_ca_and_sequence(pdb)
        if generated_sequence != sequence or len(generated_ca) != len(sequence):
            raise ValueError(f"FASTA/PDB sequence mismatch for {fasta.stem}")
        records.append(
            {
                "id": fasta.stem,
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "generated_pdb": str(pdb.resolve()),
            }
        )
    return records


def _prediction_rows(prediction_dir: Path) -> dict[str, dict[str, str]]:
    summary = prediction_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"ESMFold did not produce {summary}")
    with summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        candidate = Path(row["pdb"])
        if not candidate.is_file():
            candidate = prediction_dir / candidate.name
        if not candidate.is_file():
            raise FileNotFoundError(
                f"missing ESMFold PDB referenced by {summary}: {row['pdb']}"
            )
        row["pdb"] = str(candidate.resolve())
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("ESMFold summary contains duplicate identifiers")
    return by_id


def atom37_masked_mean_plddt(file: Path) -> tuple[float, int]:
    """Return ESMFold pLDDT averaged over emitted Atom37 protein atoms."""

    values: list[float] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  "):
            continue
        if line[16:17] not in {"", " ", "A"}:
            continue
        try:
            value = float(line[60:66])
        except ValueError as error:
            raise ValueError(f"invalid ATOM B factor in {file}: {line!r}") from error
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"invalid ATOM pLDDT in {file}: {value!r}")
        values.append(value)
    if not values:
        raise ValueError(f"no protein ATOM pLDDT values in {file}")
    return math.fsum(values) / len(values), len(values)


def score_predictions(
    records: list[dict[str, Any]],
    prediction_dir: Path,
    *,
    variant: str,
    step: int,
    length: int,
) -> list[dict[str, Any]]:
    """Strictly score generated structures against completed ESMFold predictions."""

    predictions = _prediction_rows(prediction_dir)
    expected_ids = {str(record["id"]) for record in records}
    if set(predictions) != expected_ids:
        raise ValueError(
            "ESMFold identifiers differ from inputs: "
            f"missing={sorted(expected_ids - set(predictions))} "
            f"unexpected={sorted(set(predictions) - expected_ids)}"
        )
    rows = []
    for record in records:
        identifier = str(record["id"])
        prediction = predictions[identifier]
        predicted_pdb = Path(prediction["pdb"])
        if not predicted_pdb.is_file():
            raise FileNotFoundError(f"missing ESMFold PDB: {predicted_pdb}")
        generated_ca, generated_sequence = _pdb_ca_and_sequence(
            Path(record["generated_pdb"])
        )
        predicted_ca, predicted_sequence = _pdb_ca_and_sequence(predicted_pdb)
        sequence = str(record["sequence"])
        if generated_sequence != sequence or predicted_sequence != sequence:
            raise ValueError(f"strict sequence mismatch while scoring {identifier}")
        if len(generated_ca) != length or len(predicted_ca) != length:
            raise ValueError(f"CA length mismatch while scoring {identifier}")
        rmsd = kabsch_rmsd(generated_ca, predicted_ca)
        tm_score = residue_indexed_tm_score(generated_ca, predicted_ca)
        summary_mean_plddt = float(prediction["mean_plddt"])
        if not math.isfinite(summary_mean_plddt) or not 0.0 <= summary_mean_plddt <= 100.0:
            raise ValueError(f"invalid ESMFold summary pLDDT for {identifier}")
        mean_plddt, plddt_atom_count = atom37_masked_mean_plddt(predicted_pdb)
        rows.append(
            {
                **record,
                "variant": variant,
                "step": step,
                "length": length,
                "sample_index": int(identifier.removeprefix("sample_")),
                "esmfold_pdb": str(predicted_pdb.resolve()),
                "esmfold_summary_mean_plddt": summary_mean_plddt,
                "mean_plddt": mean_plddt,
                "mean_plddt_atom_count": plddt_atom_count,
                "mean_plddt_source": "pdb_atom37_masked_b_factors",
                "ca_rmsd_a": rmsd,
                "residue_indexed_tm_score": tm_score,
                "rmsd_lt_2": int(rmsd < DESIGNABLE_RMSD_A),
                "plddt_gt_80": int(mean_plddt > DESIGNABLE_PLDDT),
                "designable": int(
                    rmsd < DESIGNABLE_RMSD_A and mean_plddt > DESIGNABLE_PLDDT
                ),
                "status": "ok",
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "fold_success_count": len(rows),
        "designable_count": sum(int(row["designable"]) for row in rows),
        "designability_rate": float(np.mean([row["designable"] for row in rows])),
        "rmsd_lt_2_rate": float(np.mean([row["rmsd_lt_2"] for row in rows])),
        "plddt_gt_80_rate": float(np.mean([row["plddt_gt_80"] for row in rows])),
        "mean_ca_rmsd_a": float(np.mean([row["ca_rmsd_a"] for row in rows])),
        "median_ca_rmsd_a": float(np.median([row["ca_rmsd_a"] for row in rows])),
        "mean_plddt": float(np.mean([row["mean_plddt"] for row in rows])),
        "mean_residue_indexed_tm_score": float(
            np.mean([row["residue_indexed_tm_score"] for row in rows])
        ),
    }


def _write_csv(file: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    temporary = file.with_suffix(file.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--expected-count", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Reuse existing ESMFold predictions instead of running the wrapper.",
    )
    parser.add_argument(
        "--wrapper",
        type=Path,
        default=Path(
            "/mnt/gbi-shared/home/kiarash-jamali/bin/esmfold_predict_container"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    records = _sample_records(
        args.sample_dir,
        expected_count=args.expected_count,
        limit=args.limit,
    )
    if any(len(record["sequence"]) != args.length for record in records):
        raise ValueError("input sequence length differs from --length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_fasta = args.output_dir / "input.fasta"
    _atomic_write_text(
        input_fasta,
        "".join(f">{record['id']}\n{record['sequence']}\n" for record in records),
    )
    command = None
    if args.prediction_dir is not None:
        prediction_dir = args.prediction_dir.resolve()
        if not prediction_dir.is_dir():
            raise FileNotFoundError(f"missing ESMFold prediction directory: {prediction_dir}")
    else:
        if not args.wrapper.is_file():
            raise FileNotFoundError(f"missing ESMFold wrapper: {args.wrapper}")
        prediction_dir = args.output_dir / "predictions"
        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)
        prediction_dir.mkdir()
        command = [
            str(args.wrapper),
            str(input_fasta),
            "--outdir",
            str(prediction_dir),
            "--chunk-size",
            str(args.chunk_size),
        ]
        if args.bf16:
            command.append("--bf16")
        subprocess.run(command, check=True)

    rows = score_predictions(
        records,
        prediction_dir,
        variant=args.variant,
        step=args.step,
        length=args.length,
    )
    _atomic_write_json(args.output_dir / "per_sample.json", rows)
    _write_csv(args.output_dir / "per_sample.csv", rows)
    summary = {
        "variant": args.variant,
        "step": args.step,
        "length": args.length,
        "thresholds": {
            "ca_rmsd_a": DESIGNABLE_RMSD_A,
            "mean_plddt": DESIGNABLE_PLDDT,
            "comparison": "ca_rmsd_a < threshold and mean_plddt > threshold",
        },
        "input_records": records,
        "esmfold_command": command,
        "prediction_dir": str(prediction_dir),
        "mean_plddt_source": "pdb_atom37_masked_b_factors",
        "metrics": _summary(rows),
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
