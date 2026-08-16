from collections import Counter
from pathlib import Path
import sys

_koochak_root = Path(__file__).parents[1] / "external" / "koochak"
if "koochak" in sys.modules and not Path(sys.modules["koochak"].__file__).resolve().is_relative_to(_koochak_root.resolve()):
    for _module_name in tuple(sys.modules):
        if _module_name == "koochak" or _module_name.startswith("koochak."):
            del sys.modules[_module_name]

from scripts.analyze_sequence_diversity_16x100k import _bootstrap, _summaries  # noqa: E402
from scripts.submit_local_center_sequence_diversity_16x100k import (  # noqa: E402
    VARIANTS,
    _prepare_tasks,
)


def test_campaign_variants_are_exactly_the_reported_factorial() -> None:
    assert len(VARIANTS) == 16
    assert len({variant.name for variant in VARIANTS}) == 16
    assert VARIANTS[0].name == "hard05_uniform_nojs"
    assert VARIANTS[0].sigma_max == 0.5
    assert VARIANTS[0].ramp_max is None
    assert VARIANTS[-1].name == "lin05to20_polar2_js005"
    assert VARIANTS[-1].ramp_max == 2.0
    assert VARIANTS[-1].polar_weight == 2.0
    assert VARIANTS[-1].marginal_js_weight == 0.05


def test_campaign_dag_has_exact_task_counts_and_unique_paths() -> None:
    workflow, tasks = _prepare_tasks("908ae771b9288512dc4c7ae5ce9c2f6ec73e8037")
    assert workflow == "hk-local-center-seq-16x100k-908ae77-v1"
    assert len(tasks) == 104
    assert len({task["task_id"] for task in tasks}) == 104
    assert Counter(task["resource"] for task in tasks) == Counter(
        {"preflight": 16, "canary": 2, "cpu": 6, "train": 16, "sample": 64}
    )
    root = "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/local-center-seq-16x100k/908ae771b9288512dc4c7ae5ce9c2f6ec73e8037/"
    assert all(task["run"].run_dir.startswith(root) for task in tasks)


def test_analysis_summaries_and_paired_bootstrap_are_deterministic() -> None:
    variants = [variant.name for variant in VARIANTS]
    rows = []
    for variant in variants:
        for length in (64, 96, 128):
            for index in range(2):
                sequence = "A" * length if index == 0 else "R" * length
                rows.append({
                    "variant": variant,
                    "length": length,
                    "index": index,
                    "sequence": sequence,
                    "effective_alphabet": 1.0,
                    "entropy_bits": 0.0,
                    "max_residue_fraction": 1.0,
                    "max_homopolymer_run": float(length),
                    "ca_step_bad_fraction": 0.0,
                    "ca_clashes_per_residue": 0.0,
                })
    summaries = _summaries(rows)
    assert summaries["hard05_uniform_nojs"]["unique_sequence_fraction_macro_average"] == 1.0
    first = _bootstrap(rows, draws=20)
    second = _bootstrap(rows, draws=20)
    assert first == second
