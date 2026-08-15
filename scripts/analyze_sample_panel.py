#!/usr/bin/env python3
"""Summarize fixed-panel sequence and coarse geometry diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def _sequence(file: Path) -> str:
    return "".join(line.strip() for line in file.read_text().splitlines() if not line.startswith(">"))


def _sequence_metrics(sequence: str) -> dict[str, float]:
    counts = {letter: sequence.count(letter) for letter in "ACDEFGHIKLMNPQRSTVWY"}
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values() if count]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    longest = current = 0
    previous = ""
    for letter in sequence:
        current = current + 1 if letter == previous else 1
        longest = max(longest, current)
        previous = letter
    return {
        "effective_alphabet": 2.0**entropy,
        "entropy_bits": entropy,
        "max_residue_fraction": max(counts.values(), default=0) / max(total, 1),
        "max_homopolymer_run": float(longest),
    }


def _ca_coordinates(file: Path) -> list[tuple[float, float, float]]:
    coordinates: list[tuple[float, float, float]] = []
    for line in file.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coordinates


def _geometry_metrics(file: Path) -> dict[str, float]:
    points = _ca_coordinates(file)
    if len(points) < 2:
        return {"ca_step_bad_fraction": 1.0, "ca_clashes_per_residue": 1.0}
    steps = [math.dist(left, right) for left, right in zip(points, points[1:])]
    bad_steps = sum(abs(distance - 3.8) > 0.6 for distance in steps)
    clashes = 0
    for index, left in enumerate(points):
        for other_index in range(index + 3, len(points)):
            if math.dist(left, points[other_index]) < 3.0:
                clashes += 1
    return {
        "ca_step_bad_fraction": bad_steps / len(steps),
        "ca_clashes_per_residue": clashes / len(points),
    }


def _summarize(directory: Path) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    for fasta in sorted(directory.glob("L*/sample_*.fasta")):
        pdb = fasta.with_suffix(".pdb")
        if not pdb.exists():
            continue
        row = _sequence_metrics(_sequence(fasta))
        row.update(_geometry_metrics(pdb))
        row["length"] = float(len(_sequence(fasta)))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no paired FASTA/PDB samples found below {directory}")
    lengths = sorted({int(row["length"]) for row in rows})
    metrics = [key for key in rows[0] if key != "length"]
    by_length = {
        str(length): {key: mean(row[key] for row in rows if row["length"] == length) for key in metrics}
        for length in lengths
    }
    return {
        "sample_count": len(rows),
        "lengths": lengths,
        "aggregate": {key: mean(row[key] for row in rows) for key in metrics},
        "by_length": by_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = {}
    for directory in sorted(path for path in args.panel.iterdir() if path.is_dir()):
        variants[directory.name] = _summarize(directory)
    result = {"panel": str(args.panel), "variants": variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
