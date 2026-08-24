#!/usr/bin/env python3
"""Reduce canonical-pLDDT history audits into ledger-ready comparisons."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean, median
from typing import Any, Iterable

from scripts.esmfold_rescore_factorials import legacy_quadrature_groups, old_kaveh_factorial


def _read_jsonl(file: Path) -> list[dict[str, Any]]:
    with file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    matched = [row for row in selected if row["ca_rmsd_a"] is not None]
    if not selected:
        raise ValueError("cannot summarize an empty panel")
    result: dict[str, Any] = {
        "prediction_count": len(selected),
        "matched_count": len(matched),
        "original_mean_plddt": fmean(float(row["original_mean_plddt"]) for row in selected),
        "corrected_mean_plddt": fmean(float(row["corrected_mean_plddt"]) for row in selected),
        "original_plddt_gt_80_count": sum(float(row["original_mean_plddt"]) > 80 for row in selected),
        "corrected_plddt_gt_80_count": sum(float(row["corrected_mean_plddt"]) > 80 for row in selected),
    }
    if matched:
        old = sum(int(row["old_designable"]) for row in matched)
        new = sum(int(row["new_designable"]) for row in matched)
        result.update(
            {
                "rmsd_lt_2_count": sum(int(row["rmsd_lt_2"]) for row in matched),
                "old_designable_count": old,
                "new_designable_count": new,
                "old_designability_rate": old / len(matched),
                "new_designability_rate": new / len(matched),
                "new_designability_wilson95": _wilson(new, len(matched)),
                "median_ca_rmsd_a": median(float(row["ca_rmsd_a"]) for row in matched),
            }
        )
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _paired_bootstrap(
    keyed: dict[str, dict[tuple[int, str], int]], *, reference: str, draws: int, seed: int
) -> dict[str, Any]:
    keys = sorted(keyed[reference])
    if any(set(values) != set(keys) for values in keyed.values()):
        raise ValueError("paired panels differ")
    rng = random.Random(seed)
    credits = {label: 0.0 for label in keyed}
    differences: dict[str, list[float]] = {label: [] for label in keyed if label != reference}
    for _ in range(draws):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        rates = {label: fmean(values[key] for key in sampled) for label, values in keyed.items()}
        maximum = max(rates.values())
        winners = [label for label, value in rates.items() if value == maximum]
        for label in winners:
            credits[label] += 1 / len(winners)
        for label, values in differences.items():
            values.append(rates[label] - rates[reference])
    return {
        "draws": draws,
        "seed": seed,
        "reference": reference,
        "probability_best": {label: value / draws for label, value in credits.items()},
        "difference_vs_reference": {
            label: {
                "mean": fmean(values),
                "paired_bootstrap95": [_quantile(values, 0.025), _quantile(values, 0.975)],
            }
            for label, values in differences.items()
        },
    }


def _length128(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pattern = re.compile(r"/shards/step(\d+)/([^/]+)/L(\d+)/predictions/")
    groups: dict[tuple[int, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row["campaign"] != "delayed-sidechain-offset-length128-esmfold":
            continue
        match = pattern.search(row["panel"])
        if match and row["ca_rmsd_a"] is not None:
            groups[(int(match[1]), match[2])].append((int(match[3]), row))
    summarized: dict[str, Any] = {}
    for (step, arm), items in groups.items():
        summarized.setdefault(str(step), {})[arm] = {
            "aggregate": _metrics(row for _, row in items),
            "by_length": {
                str(length): _metrics(row for item_length, row in items if item_length == length)
                for length in sorted({length for length, _ in items})
            },
        }
    final_items = {arm: items for (step, arm), items in groups.items() if step == 200_000}
    if not final_items:
        return {"steps": summarized, "final_ranking": [], "final_bootstrap": None}
    keyed = {
        arm: {(length, str(row["id"])): int(row["new_designable"]) for length, row in items}
        for arm, items in final_items.items()
    }
    ranking = sorted(
        final_items,
        key=lambda arm: (
            -summarized["200000"][arm]["aggregate"]["new_designability_rate"],
            summarized["200000"][arm]["aggregate"]["median_ca_rmsd_a"],
        ),
    )
    return {
        "steps": summarized,
        "final_ranking": ranking,
        "final_bootstrap": _paired_bootstrap(
            keyed, reference="ratio1_control", draws=10_000, seed=20260820
        ),
    }


def _panel_tables(
    panels: list[dict[str, Any]], rows_by_panel: dict[tuple[str, str], list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaigns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique: dict[str, dict[str, Any]] = {}
    for panel in panels:
        if panel.get("status") != "ok" or panel.get("canary") or not panel["matched_sample_count"]:
            continue
        key = (panel["audit_root"], panel["panel"])
        summary = {
            "root": panel["audit_root"],
            "panel": panel["panel"],
            "campaign": panel["campaign"],
            **_metrics(rows_by_panel[key]),
        }
        signature = str(panel["panel_signature"])
        if not any(item["panel_signature"] == signature for item in campaigns[panel["campaign"]]):
            campaigns[panel["campaign"]].append({"panel_signature": signature, **summary})
        unique.setdefault(signature, summary)
    campaign_summary = {
        campaign: {
            "panel_count": len(items),
            "prediction_count": sum(item["prediction_count"] for item in items),
            "old_designable_count": sum(item["old_designable_count"] for item in items),
            "new_designable_count": sum(item["new_designable_count"] for item in items),
            "best_panel": max(items, key=lambda item: (item["new_designability_rate"], item["matched_count"])),
            "panels": sorted(
                items,
                key=lambda item: (item["new_designability_rate"], item["matched_count"]),
                reverse=True,
            ),
        }
        for campaign, items in campaigns.items()
    }
    leaderboard = sorted(
        unique.values(), key=lambda item: (item["new_designability_rate"], item["matched_count"]), reverse=True
    )
    return campaign_summary, leaderboard


def _mpnn(rows: list[dict[str, Any]], metadata_root: Path) -> dict[str, Any]:
    pattern = re.compile(r"step-scale-mpnn-saveability/.+/folds/(scale[^/]+)/predictions/")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        match = pattern.search(row["panel"])
        if match:
            groups[match[1]].append(row)
    result = {}
    for scale, panel_rows in groups.items():
        with (metadata_root / scale / "redesign_metadata.csv").open(newline="", encoding="utf-8") as handle:
            metadata_rows = list(csv.DictReader(handle))
        metadata = {int(row["sample_index"]): int(row["source_sample_index"]) for row in metadata_rows}
        source_match = re.search(
            r"current-sampler-step-scale-designability/([0-9a-f]{40})/",
            metadata_rows[0]["source_pdb"],
        )
        if source_match is None:
            raise ValueError(f"cannot identify ProteinMPNN source panel: {scale}")
        source_commit = source_match[1]
        source_pattern = f"/{source_commit}/"
        direct_rows = [
            row
            for row in rows
            if row["campaign"] == "current-sampler-step-scale-designability"
            and source_pattern in row["panel"]
            and f"/folds/recycling_batch256_legacy/{scale}/" in row["panel"]
            and int(re.search(r"(\d+)$", str(row["id"]))[1]) < 32
        ]
        if len(direct_rows) != 32:
            raise ValueError(f"incomplete ProteinMPNN source panel: {scale}")
        by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in panel_rows:
            sample_index = int(re.search(r"(\d+)$", str(row["id"]))[1])
            by_source[metadata[sample_index]].append(row)
        if set(by_source) != set(range(32)) or any(len(items) != 4 for items in by_source.values()):
            raise ValueError(f"unbalanced ProteinMPNN panel: {scale}")
        best = {
            source_index: int(any(int(row["new_designable"]) for row in items))
            for source_index, items in by_source.items()
        }
        direct = {
            int(re.search(r"(\d+)$", str(row["id"]))[1]): int(row["new_designable"])
            for row in direct_rows
        }
        successes = sum(best.values())
        result[scale] = {
            "direct_same_backbones": _metrics(direct_rows),
            "per_sequence": _metrics(panel_rows),
            "best_of_4": {
                "backbone_count": 32,
                "designable_backbone_count": successes,
                "designability_rate": successes / 32,
                "wilson95": _wilson(successes, 32),
                "rescued_count": sum(not direct[index] and best[index] for index in direct),
                "retained_count": sum(direct[index] and best[index] for index in direct),
                "lost_count": sum(direct[index] and not best[index] for index in direct),
            },
        }
    return result


def analyze(bundles: list[Path], metadata_root: Path) -> dict[str, Any]:
    audits, rows = [], []
    for bundle in bundles:
        audit = json.loads((bundle / "audit.json").read_text(encoding="utf-8"))
        if audit["error_panel_count"]:
            raise ValueError(f"audit has errors: {bundle}")
        for panel in audit["panels"]:
            panel["audit_root"] = audit["root"]
        bundle_rows = _read_jsonl(bundle / "rows.jsonl")
        for row in bundle_rows:
            row["audit_root"] = audit["root"]
        audits.append(audit)
        rows.extend(bundle_rows)
    panels = [panel for audit in audits for panel in audit["panels"]]
    rows_by_panel: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_panel[(row["audit_root"], row["panel"])].append(row)
    campaign_summary, leaderboard = _panel_tables(panels, rows_by_panel)
    signature_counts: dict[str, int] = defaultdict(int)
    unique_rows: list[dict[str, Any]] = []
    for panel in panels:
        if panel.get("status") != "ok":
            continue
        signature = str(panel["panel_signature"])
        signature_counts[signature] += 1
        if signature_counts[signature] == 1:
            unique_rows.extend(rows_by_panel[(panel["audit_root"], panel["panel"])])
    return {
        "audit_roots": [audit["root"] for audit in audits],
        "summary_file_count": sum(audit["summary_file_count"] for audit in audits),
        "esmfold_panel_count": sum(audit["esmfold_panel_count"] for audit in audits),
        "prediction_count": sum(audit["prediction_count"] for audit in audits),
        "unique_esmfold_panel_count": len(signature_counts),
        "unique_prediction_count": len(unique_rows),
        "duplicate_panel_group_count": sum(count > 1 for count in signature_counts.values()),
        "campaigns": campaign_summary,
        "unique_panel_leaderboard": leaderboard,
        "length128_offset": _length128(unique_rows),
        "proteinmpnn": _mpnn(unique_rows, metadata_root),
        "old_kaveh_factorial": old_kaveh_factorial(unique_rows),
        "legacy_quadrature_training": legacy_quadrature_groups(unique_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--mpnn-metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.bundle, args.mpnn_metadata_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("esmfold_panel_count", "prediction_count")}, indent=2))


if __name__ == "__main__":
    main()
