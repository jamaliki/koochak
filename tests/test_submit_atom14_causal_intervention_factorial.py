from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.submit_atom14_causal_intervention_factorial as launcher


def _parent_config(tmp_path: Path, architecture: str) -> Path:
    model = {
        "node_dim": 768,
        "condition_dim": 256,
        "pair_dim": 64,
        "atom_dim": 128,
        "attention_heads": 12,
        "attention_head_dim": 64,
        "atom_heads": 4,
        "atom_head_dim": 32,
        "atom_encoder_depth": 1,
        "residue_encoder_depth": 4,
        "coarse_depth": 6,
        "residue_decoder_depth": 4,
        "atom_decoder_depth": 1,
        "patchify_mode": launcher.ARCHITECTURE_KEYS[architecture]["model.patchify_mode"],
        "coarse_pair_position": launcher.ARCHITECTURE_KEYS[architecture]["model.coarse_pair_position"],
        "coarse_pair_transition": launcher.ARCHITECTURE_KEYS[architecture]["model.coarse_pair_transition"],
    }
    config = {
        "model": model,
        "data": {
            "metadata_path": str(launcher.METADATA), "min_length": 32, "max_length": 128,
            "mean_plddt_min": 80.0, "loop_length_max": 15, "loop_content_max": 0.4,
            "packing_density_min": 0.3, "batch_size": 256, "num_workers": 8,
            "prefetch_factor": 1, "length_buckets": [64, 96, 128],
            "patch_capacities": [16, 24, 32], "pin_memory": True,
            "persistent_workers": True, "shard_cache_size": None,
        },
        "diffusion": {"p_mean": -1.2, "p_std": 1.5},
        "loss": {"aatype_sigma_max": 0.5, "aatype_sigma_ramp_max": None},
        "train": {
            "max_steps": 500_000, "ckpt_every": 50_000, "keep_last_k": 12,
            "ddp": False, "grad_accum": 1, "amp": "bf16",
            "self_conditioning_probability": 0.5,
            "prefetch_batches": 2, "prefetch_threaded": True, "prefetch_pipeline": "two_stage",
            "compile": {"enabled": True}, "require_compile": True, "require_fused": True,
        },
        "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1.0e-8},
        "logging": {"csv_path": "/parent/log.csv", "jsonl_path": "/parent/log.jsonl"},
    }
    file = tmp_path / f"{architecture}.yaml"
    OmegaConf.save(OmegaConf.create(config), file)
    return file


def _parents(tmp_path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for architecture in launcher.ARCHITECTURES:
        file = _parent_config(tmp_path, architecture)
        result[architecture] = {
            "config": file,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    return result


def _manifest(run) -> dict[str, object]:
    prepared = run.run if hasattr(run, "run") else run
    artifact = next(item for item in prepared.artifacts if item.path.endswith("launch.json"))
    return json.loads(artifact.content)


def test_builds_exact_16_cell_82_task_dag_with_robust_recovery(tmp_path: Path) -> None:
    workflow, diffs = launcher.build_workflow(
        "a" * 40, parent_cells=_parents(tmp_path), output_root=tmp_path / "output"
    )
    counts = Counter(task.task_id.split("-", 1)[0] for task in workflow.tasks)
    assert len(launcher.CELLS) == 16
    assert len({cell.cell_id for cell in launcher.CELLS}) == 16
    assert len(workflow.tasks) == 82
    assert counts == Counter({"preflight": 16, "train": 16, "sample": 16, "esmfold": 16, "analysis": 16, "attest": 1, "aggregate": 1})
    assert len(diffs) == 16
    assert all(
        task.to_scruffy_spec(
            request_id=workflow.request_id,
            workflow_id=workflow.workflow_id,
            project_id=workflow.project_id,
        )["recovery"] == launcher.RECOVERY
        for task in workflow.tasks
    )


def test_factor_off_cells_do_not_materialize_factor_keys_and_on_cells_patch_exactly(tmp_path: Path) -> None:
    workflow, diffs = launcher.build_workflow(
        "b" * 40, parent_cells=_parents(tmp_path), output_root=tmp_path / "output"
    )
    by_id = {item["cell_id"]: item for item in diffs}
    off = by_id[f"{launcher.ARCHITECTURES[0]}-o0r0t0"]
    assert {item["path"] for item in off["differences"]} == launcher.OPERATIONAL_PATHS | launcher.OUTPUT_PATHS
    all_on = by_id[f"{launcher.ARCHITECTURES[0]}-o1r1t1"]
    assert {item["path"] for item in all_on["differences"]} == launcher.OPERATIONAL_PATHS | launcher.OUTPUT_PATHS | launcher.FACTOR_PATHS

    train_tasks = {task.task_id.removeprefix("train-"): task for task in workflow.tasks if task.task_id.startswith("train-")}
    off_config = launcher._config_container(train_tasks[off["cell_id"]].run)
    assert all(launcher._at(off_config, key, launcher._MISSING) is launcher._MISSING for key in launcher.FACTOR_PATHS)
    on_config = launcher._config_container(train_tasks[all_on["cell_id"]].run)
    assert launcher._at(on_config, "loss.smooth_lddt_sigma_max") == 3.0
    assert launcher._at(on_config, "model.atom_to_residue_transport") == "backbone_first"
    assert not any("threshold" in patch.path for patch in launcher._factor_patches(launcher.CELLS[-1]))


def test_preflight_command_is_objective_specific_and_exactly_one_checkpoint_is_evaluated(tmp_path: Path) -> None:
    workflow, _ = launcher.build_workflow(
        "c" * 40, parent_cells=_parents(tmp_path), output_root=tmp_path / "output"
    )
    preflights = {task.task_id.removeprefix("preflight-"): task for task in workflow.tasks if task.task_id.startswith("preflight-")}
    objective_id = next(cell.cell_id for cell in launcher.CELLS if cell.objective)
    control_id = next(cell.cell_id for cell in launcher.CELLS if not cell.objective)
    objective_argv = _manifest(preflights[objective_id])["argv"]
    control_argv = _manifest(preflights[control_id])["argv"]
    assert "--objective-repair" in objective_argv
    assert "--objective-repair" not in control_argv
    assert "--run-training-gate" in objective_argv
    assert "--run-training-gate" in control_argv
    assert launcher.PREFLIGHT_STEPS == 64
    assert launcher.CHECKPOINT_STEP == launcher.MAX_STEPS == 50_000
    assert launcher.TRAIN_CHECKPOINT_INTERVAL == 10_000
    sample_task = next(task for task in workflow.tasks if task.task_id.startswith("sample-"))
    assert "step000050000" in " ".join(_manifest(sample_task.run)["argv"])


def test_resolved_diff_fails_closed_on_unapproved_change(tmp_path: Path) -> None:
    parents = _parents(tmp_path)
    parent_file = Path(parents[launcher.ARCHITECTURES[0]]["config"])
    child = OmegaConf.to_container(OmegaConf.load(parent_file), resolve=True)
    child = copy.deepcopy(child)
    child["optimizer"]["lr"] = 0.002
    with pytest.raises(AssertionError, match="unexpected resolved-config differences"):
        launcher._resolved_diff(parent_file, child, launcher.CELLS[0])


def test_required_path_check_reports_only_missing_paths(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.mkdir()
    missing = tmp_path / "missing"
    assert launcher._missing_paths((present, missing)) == [str(missing)]
