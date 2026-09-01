#!/usr/bin/env python3
"""Aggregate ESMFold self-consistency shards and rank delayed-clock variants."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, median
from typing import Any


RATE_KEYS = ("designable", "rmsd_lt_2", "plddt_gt_80")


def _atomic_write(file: Path, value: str) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, file)


def _parse_csv(value: str, cast=str) -> tuple[Any, ...]:
    result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("comma-separated values must be non-empty and unique")
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1 - probability) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(rows)}
    for key in RATE_KEYS:
        successes = sum(int(row[key]) for row in rows)
        result[f"{key}_count"] = successes
        result[f"{key}_rate"] = successes / len(rows)
        result[f"{key}_wilson95"] = _wilson(successes, len(rows))
    result.update(
        {
            "mean_ca_rmsd_a": mean(float(row["ca_rmsd_a"]) for row in rows),
            "median_ca_rmsd_a": median(float(row["ca_rmsd_a"]) for row in rows),
            "mean_plddt": mean(float(row["mean_plddt"]) for row in rows),
            "median_plddt": median(float(row["mean_plddt"]) for row in rows),
            "mean_residue_indexed_tm_score": mean(
                float(row["residue_indexed_tm_score"]) for row in rows
            ),
        }
    )
    return result


def _paired_bootstrap(
    rows_by_variant: dict[str, dict[tuple[int, int], dict[str, Any]]],
    *,
    reference: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if reference not in rows_by_variant:
        raise ValueError(f"reference variant {reference!r} is not in the panel")
    keys = sorted(rows_by_variant[reference])
    if any(set(rows) != set(keys) for rows in rows_by_variant.values()):
        raise ValueError("variants do not share the same paired length/sample panel")
    generator = random.Random(seed)
    variants = tuple(rows_by_variant)
    best_credits = {variant: 0.0 for variant in variants}
    differences = {variant: [] for variant in variants if variant != reference}
    for _draw in range(draws):
        selected = [keys[generator.randrange(len(keys))] for _index in keys]
        rates = {
            variant: mean(
                int(rows_by_variant[variant][key]["designable"]) for key in selected
            )
            for variant in variants
        }
        maximum = max(rates.values())
        winners = [variant for variant, value in rates.items() if value == maximum]
        for variant in winners:
            best_credits[variant] += 1 / len(winners)
        for variant in differences:
            differences[variant].append(rates[variant] - rates[reference])
    return {
        "draws": draws,
        "seed": seed,
        "reference": reference,
        "probability_best": {
            variant: best_credits[variant] / draws for variant in variants
        },
        "designability_difference_vs_reference": {
            variant: {
                "mean": mean(values),
                "paired_bootstrap95": [
                    _quantile(values, 0.025),
                    _quantile(values, 0.975),
                ],
            }
            for variant, values in differences.items()
        },
    }


def analyze_milestone(
    input_root: Path,
    *,
    step: int,
    variants: tuple[str, ...],
    lengths: tuple[int, ...],
    expected_per_length: int,
    sample_analysis: Path | None,
    reference: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows_by_variant: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    summaries = {}
    for variant in variants:
        rows = []
        keyed = {}
        for length in lengths:
            shard_file = input_root / variant / f"L{length:04d}" / "per_sample.json"
            shard_rows = json.loads(shard_file.read_text())
            if len(shard_rows) != expected_per_length:
                raise ValueError(
                    f"expected {expected_per_length} rows in {shard_file}, found {len(shard_rows)}"
                )
            for row in shard_rows:
                if (
                    row["variant"] != variant
                    or int(row["step"]) != step
                    or int(row["length"]) != length
                    or row["status"] != "ok"
                ):
                    raise ValueError(f"shard contract mismatch in {shard_file}")
                key = (length, int(row["sample_index"]))
                if key in keyed:
                    raise ValueError(f"duplicate sample key {key} for {variant}")
                keyed[key] = row
                rows.append(row)
        rows_by_variant[variant] = keyed
        summaries[variant] = {
            "aggregate": _metrics(rows),
            "by_length": {
                str(length): _metrics(
                    [row for row in rows if int(row["length"]) == length]
                )
                for length in lengths
            },
        }

    ranking = sorted(
        variants,
        key=lambda variant: (
            -summaries[variant]["aggregate"]["designable_rate"],
            summaries[variant]["aggregate"]["median_ca_rmsd_a"],
            -summaries[variant]["aggregate"]["mean_plddt"],
        ),
    )
    quality = None
    if sample_analysis is not None:
        quality = json.loads(sample_analysis.read_text())["variants"]
        if set(quality) != set(variants):
            raise ValueError(
                "sample-analysis variants differ from designability variants"
            )
    return {
        "step": step,
        "lengths": list(lengths),
        "samples_per_length": expected_per_length,
        "reference": reference,
        "primary_endpoint": "designable = CA RMSD < 2 A and mean pLDDT > 80",
        "ranking": list(ranking),
        "variants": summaries,
        "paired_bootstrap": _paired_bootstrap(
            rows_by_variant,
            reference=reference,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        ),
        "sample_quality": quality,
    }


def _milestone_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# ESMFold designability at step {result['step']:,}",
        "",
        "Primary endpoint: strict CA self-consistency RMSD <2 A and mean ESMFold pLDDT >80.",
        "",
        "| Rank | Variant | Designable | RMSD<2 | pLDDT>80 | Median RMSD (A) | Mean pLDDT | P(best) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    probabilities = result["paired_bootstrap"]["probability_best"]
    for index, variant in enumerate(result["ranking"], start=1):
        metrics = result["variants"][variant]["aggregate"]
        lines.append(
            f"| {index} | `{variant}` | {metrics['designable_rate']:.1%} | "
            f"{metrics['rmsd_lt_2_rate']:.1%} | {metrics['plddt_gt_80_rate']:.1%} | "
            f"{metrics['median_ca_rmsd_a']:.3f} | {metrics['mean_plddt']:.2f} | "
            f"{probabilities[variant]:.1%} |"
        )
    lines.extend(
        [
            "",
            "`P(best)` is a deterministic paired-bootstrap selection probability. "
            f"Paired differences use `{result['reference']}` as the reference; "
            "overlapping uncertainty should not be reported as a definitive winner.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_campaign(
    root: Path,
    milestones: tuple[int, ...],
    *,
    title: str = "ESMFold designability campaign",
) -> dict[str, Any]:
    documents = {
        step: json.loads((root / f"milestone_step{step:06d}.json").read_text())
        for step in milestones
    }
    variants = tuple(documents[milestones[0]]["variants"])
    if any(
        set(document["variants"]) != set(variants) for document in documents.values()
    ):
        raise ValueError("milestone variant sets differ")
    trajectories = {
        variant: {
            str(step): documents[step]["variants"][variant]["aggregate"]
            for step in milestones
        }
        for variant in variants
    }
    stability_ranking = sorted(
        variants,
        key=lambda variant: (
            -mean(
                trajectories[variant][str(step)]["designable_rate"]
                for step in milestones
            )
        ),
    )
    return {
        "title": title,
        "milestones": list(milestones),
        "primary_decision_milestone": milestones[-1],
        "final_ranking": documents[milestones[-1]]["ranking"],
        "final_paired_bootstrap": documents[milestones[-1]]["paired_bootstrap"],
        "mean_milestone_designability_ranking": stability_ranking,
        "trajectories": trajectories,
    }


def _campaign_markdown(result: dict[str, Any]) -> str:
    steps = result["milestones"]
    lines = [
        f"# {result['title']}",
        "",
        f"Primary decision point: step {result['primary_decision_milestone']:,}.",
        "",
        "| Final rank | Variant | "
        + " | ".join(f"{step // 1000}k" for step in steps)
        + " |",
        "| ---: | --- | " + " | ".join("---:" for _step in steps) + " |",
    ]
    for index, variant in enumerate(result["final_ranking"], start=1):
        rates = [
            result["trajectories"][variant][str(step)]["designable_rate"]
            for step in steps
        ]
        lines.append(
            f"| {index} | `{variant}` | "
            + " | ".join(f"{rate:.1%}" for rate in rates)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    milestone_parser = subparsers.add_parser("milestone")
    milestone_parser.add_argument("--input-root", type=Path, required=True)
    milestone_parser.add_argument("--step", type=int, required=True)
    milestone_parser.add_argument("--variants", required=True)
    milestone_parser.add_argument("--lengths", required=True)
    milestone_parser.add_argument("--expected-per-length", type=int, required=True)
    milestone_parser.add_argument("--sample-analysis", type=Path)
    milestone_parser.add_argument("--reference", required=True)
    milestone_parser.add_argument("--output", type=Path, required=True)
    milestone_parser.add_argument("--report", type=Path)
    milestone_parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    milestone_parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    campaign_parser = subparsers.add_parser("campaign")
    campaign_parser.add_argument("--root", type=Path, required=True)
    campaign_parser.add_argument("--milestones", required=True)
    campaign_parser.add_argument("--output", type=Path, required=True)
    campaign_parser.add_argument("--report", type=Path)
    campaign_parser.add_argument("--title", default="ESMFold designability campaign")
    args = parser.parse_args()

    if args.command == "milestone":
        result = analyze_milestone(
            args.input_root,
            step=args.step,
            variants=_parse_csv(args.variants),
            lengths=_parse_csv(args.lengths, int),
            expected_per_length=args.expected_per_length,
            sample_analysis=args.sample_analysis,
            reference=args.reference,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        report = _milestone_markdown(result)
    else:
        result = analyze_campaign(
            args.root,
            _parse_csv(args.milestones, int),
            title=args.title,
        )
        report = _campaign_markdown(result)
    _atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.report is not None:
        _atomic_write(args.report, report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
