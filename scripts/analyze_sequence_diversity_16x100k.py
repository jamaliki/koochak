#!/usr/bin/env python3
"""Validate and analyze the fixed-sampler panels for the 16-arm campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import random
from statistics import fmean

ALPHABET = "ARNDCQEGHILKMFPSTWYV"
EXPECTED_LENGTHS = (64, 96, 128)
EXPECTED_SAMPLING = {
    "num_steps": 200, "p_mean": -1.2, "p_std": 1.5, "gamma": 0.2,
    "churn_tmin": 0.01, "churn_tmax": 1.0, "noise_scale": 1.003,
    "step_scale": 2.25, "translation_std": 1.0, "sequence_temperature": 0.1,
}
CONTROL = "hard05_uniform_nojs"
METRICS = (
    "effective_alphabet", "entropy_bits", "max_residue_fraction",
    "max_homopolymer_run", "ca_step_bad_fraction", "ca_clashes_per_residue",
)


def _sequence(file: Path) -> str:
    return "".join(line.strip() for line in file.read_text().splitlines() if not line.startswith(">"))


def _ca_coordinates(file: Path) -> list[tuple[float, float, float]]:
    return [
        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in file.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA"
    ]


def _geometry(file: Path) -> tuple[float, float]:
    points = _ca_coordinates(file)
    if len(points) < 2:
        return 1.0, 1.0
    steps = [math.dist(left, right) for left, right in zip(points, points[1:])]
    bad = sum(abs(distance - 3.8) > 0.6 for distance in steps) / len(steps)
    clashes = sum(
        math.dist(points[index], points[other]) < 3.0
        for index in range(len(points))
        for other in range(index + 3, len(points))
    ) / len(points)
    return bad, clashes


def _sequence_metrics(sequence: str) -> dict[str, float]:
    counts = Counter(sequence)
    total = len(sequence)
    probabilities = [counts[letter] / total for letter in ALPHABET if counts[letter]]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    longest = current = 0
    previous = None
    for letter in sequence:
        current = current + 1 if letter == previous else 1
        longest = max(longest, current)
        previous = letter
    return {
        "effective_alphabet": 2.0 ** entropy,
        "entropy_bits": entropy,
        "max_residue_fraction": max((counts[letter] for letter in ALPHABET), default=0) / max(total, 1),
        "max_homopolymer_run": float(longest),
    }


def _validate_manifest(directory: Path, step: int) -> dict[str, object]:
    manifest = json.loads((directory / "manifest.json").read_text())
    checks = {
        "checkpoint_step": step,
        "weights": "ema",
        "lengths": list(EXPECTED_LENGTHS),
        "samples_per_length": 32,
        "seed": 20260813,
        "precision": "bf16",
        "compiled": True,
        "recurrent_self_conditioning": True,
        "sampling": EXPECTED_SAMPLING,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{directory}/manifest.json has {key}={manifest.get(key)!r}, expected {expected!r}")
    return manifest


def _load_rows(sample_root: Path, step: int) -> list[dict[str, object]]:
    rows = []
    for variant_dir in sorted(item for item in sample_root.iterdir() if item.is_dir()):
        _validate_manifest(variant_dir, step)
        for length in EXPECTED_LENGTHS:
            length_dir = variant_dir / f"L{length:04d}"
            fasta_files = sorted(length_dir.glob("sample_*.fasta"))
            if len(fasta_files) != 32:
                raise ValueError(f"{length_dir} contains {len(fasta_files)} samples, expected 32")
            for fasta in fasta_files:
                pdb = fasta.with_suffix(".pdb")
                if not pdb.is_file():
                    raise FileNotFoundError(pdb)
                sequence = _sequence(fasta)
                if len(sequence) != length or any(letter not in ALPHABET for letter in sequence):
                    raise ValueError(f"invalid sequence in {fasta}")
                step_bad, clashes = _geometry(pdb)
                row = {"variant": variant_dir.name, "length": length, "index": int(fasta.stem.split("_")[-1]), "sequence": sequence}
                row.update(_sequence_metrics(sequence))
                row["ca_step_bad_fraction"] = step_bad
                row["ca_clashes_per_residue"] = clashes
                rows.append(row)
    if not rows or {row["variant"] for row in rows} != {
        "hard05_uniform_nojs", "hard05_uniform_js005", "hard05_polar2_nojs", "hard05_polar2_js005",
        "hard10_uniform_nojs", "hard10_uniform_js005", "hard10_polar2_nojs", "hard10_polar2_js005",
        "lin05to10_uniform_nojs", "lin05to10_uniform_js005", "lin05to10_polar2_nojs", "lin05to10_polar2_js005",
        "lin05to20_uniform_nojs", "lin05to20_uniform_js005", "lin05to20_polar2_nojs", "lin05to20_polar2_js005",
    }:
        raise ValueError("sample root must contain exactly the 16 campaign variants")
    return rows


def _mean_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def _pairwise_identity(sequences: list[str]) -> float:
    if len(sequences) < 2:
        return 1.0
    values = [sum(left == right for left, right in zip(a, b)) / len(a) for a, b in itertools.combinations(sequences, 2)]
    return fmean(values)


def _summaries(rows: list[dict[str, object]]) -> dict[str, object]:
    variants = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        by_length = {}
        for length in EXPECTED_LENGTHS:
            length_rows = [row for row in variant_rows if row["length"] == length]
            sequences = [str(row["sequence"]) for row in length_rows]
            summary = _mean_rows(length_rows)
            summary.update({
                "unique_sequence_fraction": len(set(sequences)) / len(sequences),
                "mean_pairwise_sequence_identity": _pairwise_identity(sequences),
            })
            by_length[str(length)] = summary
        frequencies = Counter("".join(str(row["sequence"]) for row in variant_rows))
        total = sum(frequencies.values())
        variants[variant] = {
            "sample_count": len(variant_rows),
            "by_length": by_length,
            "aggregate": _mean_rows(variant_rows),
            "unique_sequence_fraction_macro_average": fmean(item["unique_sequence_fraction"] for item in by_length.values()),
            "mean_pairwise_sequence_identity_macro_average": fmean(item["mean_pairwise_sequence_identity"] for item in by_length.values()),
            "amino_acid_frequencies": {letter: frequencies[letter] / total for letter in ALPHABET},
        }
    return variants


def _bootstrap(rows: list[dict[str, object]], *, seed: int = 20260813, draws: int = 2000) -> dict[str, object]:
    control = [row for row in rows if row["variant"] == CONTROL]
    contrasts = {}
    for variant in sorted({str(row["variant"]) for row in rows} - {CONTROL}):
        candidate = [row for row in rows if row["variant"] == variant]
        per_metric = {}
        for metric in METRICS:
            rng = random.Random(f"{seed}:{variant}:{metric}")
            distributions = []
            for _ in range(draws):
                length_means = []
                for length in EXPECTED_LENGTHS:
                    left = {int(row["index"]): row for row in candidate if row["length"] == length}
                    right = {int(row["index"]): row for row in control if row["length"] == length}
                    indices = sorted(left.keys() & right.keys())
                    sampled = [rng.choice(indices) for _ in indices]
                    length_means.append(fmean(float(left[index][metric]) - float(right[index][metric]) for index in sampled))
                distributions.append(fmean(length_means))
            distributions.sort()
            per_metric[metric] = {
                "mean": fmean(distributions),
                "lower_95": distributions[int(0.025 * draws)],
                "upper_95": distributions[int(0.975 * draws) - 1],
            }
        contrasts[variant] = per_metric
    return {"control": CONTROL, "draws": draws, "seed": seed, "paired_by_index_stratified_by_length": True, "metrics": contrasts}


def analyze(sample_root: Path, step: int, output: Path) -> None:
    rows = _load_rows(sample_root, step)
    audit = output.with_name(f"{output.stem}_samples.jsonl")
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "step": step,
        "sample_root": str(sample_root),
        "manifest_contract": {"weights": "ema", "precision": "bf16", "compiled": True, "lengths": list(EXPECTED_LENGTHS), "samples_per_length": 32, "seed": 20260813},
        "variants": _summaries(rows),
        "paired_bootstrap_vs_control": _bootstrap(rows),
        "audit_rows": str(audit),
        "bootstrap_limitation": "Intervals quantify fixed-panel sampling uncertainty, not training-seed uncertainty.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def finalize(analysis_dir: Path, output: Path) -> None:
    files = sorted(analysis_dir.glob("milestone_step*.json"))
    if len(files) != 4:
        raise ValueError(f"expected four milestone JSON files, found {len(files)}")
    documents = [json.loads(file.read_text()) for file in files]
    lines = [
        "# Local-center sequence-diversity 16-arm campaign",
        "",
        "Only fixed-sampler 100k panels are promotion evidence; earlier milestones are diagnostics.",
        "Bootstrap intervals are paired by sample index and stratified by length, and quantify panel uncertainty rather than training-seed uncertainty.",
        "",
        "## Effective alphabet trajectory",
        "",
        "| Step | " + " | ".join(sorted(documents[-1]["variants"])) + " |",
        "|---:|" + "---:|" * 16,
    ]
    for document in documents:
        values = [f"{document['variants'][variant]['aggregate']['effective_alphabet']:.4f}" for variant in sorted(document["variants"])]
        lines.append(f"| {document['step']} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Planned contrasts",
        "",
        "- Polar weight 2 versus 1 within every schedule/JS pair.",
        "- JS 0.05 versus 0 within every schedule/polar pair.",
        "- Each schedule versus hard05 within every polar/JS pair.",
        "- Schedule-by-polar, schedule-by-JS, and polar-by-JS interactions.",
        "- 25k-to-100k rank reversals.",
        "",
        "Promotion is guarded by entropy, composition, uniqueness, identity, geometry, clashes, and the full sequence-to-structure panel; no early milestone is an early-stop criterion.",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--analysis-dir", type=Path)
    args = parser.parse_args()
    if args.finalize:
        if args.analysis_dir is None:
            parser.error("--analysis-dir is required with --finalize")
        finalize(args.analysis_dir, args.output)
    else:
        if args.sample_root is None or args.step is None:
            parser.error("--sample-root and --step are required")
        analyze(args.sample_root, args.step, args.output)


if __name__ == "__main__":
    main()
