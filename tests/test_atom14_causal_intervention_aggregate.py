from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_atom14_causal_intervention import aggregate, main
import scripts.submit_atom14_causal_intervention_factorial as launcher


def _population(count: int) -> dict[str, object]:
    return {
        "sample_count": count,
        "pair_count": count * (count - 1) // 2,
        "pair_similarity_mean": 0.7,
        "pair_similarity_median": 0.7,
        "same_fold_pair_count": 2,
        "same_fold_pair_fraction": 0.1,
        "same_cluster_pair_count": 2,
        "same_cluster_pair_fraction": 0.1,
        "cluster_count": 2 if count else 0,
        "effective_cluster_count": 1.8 if count else 0.0,
        "largest_cluster_fraction": 0.5 if count else None,
        "cluster_sizes": [count // 2, count - count // 2] if count else [],
    }


def _analysis(cell_id: str, *, primary: int = 2, designable: int = 4) -> dict[str, object]:
    return {
        "method": "Progres protein graph embeddings",
        "progres_version": "1.1.0",
        "same_fold_threshold": 0.8,
        "designability_definition": "CA RMSD < 2 A and mean ESMFold pLDDT > 80",
        "panel": cell_id,
        "step": 50_000,
        "designable_count": designable,
        "designable_samples": [f"sample_{index:05d}" for index in range(designable)],
        "primary_metric": {
            "name": "generated.designable_subset.complete_linkage_progres_cluster_count",
            "value": primary,
        },
        "secondary_concordance_metric": {
            "name": "esmfold.designable_subset.complete_linkage_progres_cluster_count",
            "value": primary,
        },
        "primary_designable_progres_cluster_count": primary,
        "secondary_esmfold_designable_progres_cluster_count": primary,
        "generated": {
            "all_samples": _population(32),
            "designable_subset": _population(designable),
        },
        "esmfold": {
            "all_samples": _population(32),
            "designable_subset": _population(designable),
        },
        "generated_sequence_diversity": {
            "sample_count": 32,
            "unique_sequence_count": 12,
            "largest_identical_sequence_fraction": 0.2,
            "pairwise_sequence_identity_mean": 0.31,
        },
    }


def _inputs(tmp_path: Path) -> list[str]:
    values = []
    for cell in launcher.CELLS:
        file = tmp_path / f"{cell.cell_id}.json"
        file.write_text(json.dumps(_analysis(cell.cell_id)) + "\n", encoding="utf-8")
        values.append(f"{cell.cell_id}={file}")
    return values


def test_aggregate_emits_validated_cell_analyses_and_effects_table(tmp_path: Path) -> None:
    output = tmp_path / "aggregate.json"
    document = aggregate(
        _inputs(tmp_path), output=output, workflow="workflow", step=50_000
    )
    assert output.is_file()
    assert document["schema"] == "hierarchical-kaveh.atom14-causal-intervention-aggregate.v1"
    assert len(document["cell_analyses"]) == 16
    assert len(document["effects_ready"]) == 16
    assert document["effects_ready"][0]["cell_id"] == sorted(cell.cell_id for cell in launcher.CELLS)[0]
    assert document["effects_ready"][0]["primary_designable_progres_cluster_count"] == 2
    assert document["effects_ready"][0]["designability_fraction"] == 0.125
    assert "generated_designable_largest_cluster_fraction" in document["effects_ready_columns"]


def test_aggregate_cli_accepts_the_launcher_input_shape(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "cli.json"
    argv = []
    for value in inputs:
        argv.extend(["--analysis", value])
    argv.extend(["--output", str(output), "--workflow", "workflow", "--step", "50000"])
    assert main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["expected_cell_count"] == 16


def test_aggregate_rejects_nonfinite_analysis(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    file = tmp_path / f"{launcher.CELLS[0].cell_id}.json"
    document = json.loads(file.read_text(encoding="utf-8"))
    document["generated"]["designable_subset"]["effective_cluster_count"] = float("nan")
    file.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        aggregate(inputs, output=tmp_path / "bad.json", workflow="workflow")


def test_aggregate_rejects_incomplete_factorial(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)[:-1]
    with pytest.raises(ValueError, match="expected 16"):
        aggregate(inputs, output=tmp_path / "bad.json", workflow="workflow")
