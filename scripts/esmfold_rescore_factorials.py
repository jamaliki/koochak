"""Reducers for historical training-factorial ESMFold audit rows."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
import random
import re
from statistics import fmean
from typing import Any, Callable


FACTOR_NAMES = (
    "mixed_schedule",
    "deep_supervision",
    "aatype_recycling",
    "structural_recycling",
)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    successes = sum(int(row["new_designable"]) for row in rows)
    return {
        "sample_count": count,
        "old_designable_count": sum(int(row["old_designable"]) for row in rows),
        "new_designable_count": successes,
        "new_designability_rate": successes / count,
        "rmsd_lt_2_rate": fmean(int(row["rmsd_lt_2"]) for row in rows),
        "plddt_gt_80_rate": fmean(float(row["corrected_mean_plddt"]) > 80 for row in rows),
        "corrected_mean_plddt": fmean(float(row["corrected_mean_plddt"]) for row in rows),
    }


def _contrast(rates: dict[str, float], cells: list[str], predicate: Callable[[str], bool]) -> float:
    positive = [rates[cell] for cell in cells if predicate(cell)]
    negative = [rates[cell] for cell in cells if not predicate(cell)]
    return fmean(positive) - fmean(negative)


def _factorial_contrasts(rates: dict[str, float], cells: list[str]) -> dict[str, float]:
    result = {}
    variable = [index for index in range(4) if {cell[index + 1] for cell in cells} == {"0", "1"}]
    for index in variable:
        result[FACTOR_NAMES[index]] = _contrast(rates, cells, lambda cell, index=index: cell[index + 1] == "1")
    for left, right in combinations(variable, 2):
        result[f"{FACTOR_NAMES[left]}*{FACTOR_NAMES[right]}"] = _contrast(
            rates,
            cells,
            lambda cell, left=left, right=right: cell[left + 1] == cell[right + 1],
        )
    return result


def _bootstrap_factorial(
    keyed: dict[str, dict[tuple[int, str], int]], *, draws: int, seed: int
) -> dict[str, Any]:
    cells = sorted(keyed)
    keys = sorted(keyed[cells[0]])
    if any(set(values) != set(keys) for values in keyed.values()):
        raise ValueError("old-Kaveh factorial cells are not paired")
    observed_rates = {cell: fmean(keyed[cell][key] for key in keys) for cell in cells}
    observed = _factorial_contrasts(observed_rates, cells)
    samples = {name: [] for name in observed}
    rng = random.Random(seed)
    for _ in range(draws):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        rates = {cell: fmean(keyed[cell][key] for key in selected) for cell in cells}
        for name, value in _factorial_contrasts(rates, cells).items():
            samples[name].append(value)
    return {
        name: {
            "effect": observed[name],
            "paired_bootstrap95": [_quantile(values, 0.025), _quantile(values, 0.975)],
            "probability_positive": sum(value > 0 for value in values) / draws,
        }
        for name, values in samples.items()
    }


def _paired_difference(
    left: dict[tuple[int, str], int],
    right: dict[tuple[int, str], int],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(left)
    if set(right) != set(keys):
        raise ValueError("sampler panels are not paired")
    differences = [right[key] - left[key] for key in keys]
    rng = random.Random(seed)
    samples = [
        fmean(differences[rng.randrange(len(keys))] for _ in keys) for _ in range(draws)
    ]
    return {
        "difference": fmean(differences),
        "paired_bootstrap95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
        "probability_positive": sum(value > 0 for value in samples) / draws,
    }


def old_kaveh_factorial(
    rows: list[dict[str, Any]], *, draws: int = 10_000, seed: int = 20260820
) -> dict[str, Any]:
    pattern = re.compile(r"/shards/step(\d+)/([^/]+)/(f[01]{4}(?:_oldobj)?)/L(\d+)/")
    grouped: dict[tuple[int, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row["campaign"] != "old-kaveh-training-factorial-18x200k" or row["ca_rmsd_a"] is None:
            continue
        match = pattern.search(row["panel"])
        if match:
            grouped[(int(match[1]), match[2], match[3])].append((int(match[4]), row))
    panels = {
        f"{step}:{preset}:{cell}": _summary([row for _, row in items])
        for (step, preset, cell), items in grouped.items()
    }
    keyed = {
        (step, preset, cell): {
            (length, str(row["id"])): int(row["new_designable"]) for length, row in items
        }
        for (step, preset, cell), items in grouped.items()
        if re.fullmatch(r"f[01]{4}", cell)
    }
    effects = {}
    for step, preset in sorted({(step, preset) for step, preset, _ in keyed}):
        cells = {cell: values for (item_step, item_preset, cell), values in keyed.items() if (item_step, item_preset) == (step, preset)}
        effects[f"{step}:{preset}"] = _bootstrap_factorial(cells, draws=draws, seed=seed + step)
    sampler_differences = {}
    for step in sorted({step for step, _, _ in keyed}):
        for cell in sorted({cell for item_step, _, cell in keyed if item_step == step}):
            current = keyed.get((step, "current", cell))
            old_struct = keyed.get((step, "old_struct", cell))
            if current is not None and old_struct is not None:
                sampler_differences[f"{step}:{cell}"] = _paired_difference(
                    current, old_struct, draws=draws, seed=seed + step + int(cell[1:], 2)
                )
    ranking = sorted(
        panels,
        key=lambda label: (panels[label]["new_designability_rate"], panels[label]["sample_count"]),
        reverse=True,
    )
    return {
        "factor_order": list(FACTOR_NAMES),
        "panels": panels,
        "ranking": ranking,
        "factorial_effects": effects,
        "old_struct_vs_current": sampler_differences,
    }


def legacy_quadrature_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    campaigns = (
        "delayed-sidechain-legacy-quadrature-family-12x200k",
        "delayed-sidechain-legacy-quadrature-length128-expansion-15x200k",
    )
    pattern = re.compile(r"/shards/step(\d+)/length128/([^/]+)/L(\d+)/")
    result = {}
    for campaign in campaigns:
        groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["campaign"] != campaign or row["ca_rmsd_a"] is None:
                continue
            match = pattern.search(row["panel"])
            if match:
                groups[(int(match[1]), match[2])].append(row)
        panels = {f"{step}:{variant}": _summary(items) for (step, variant), items in groups.items()}
        result[campaign] = {
            "panels": panels,
            "ranking": sorted(
                panels, key=lambda label: panels[label]["new_designability_rate"], reverse=True
            ),
        }
    return result
