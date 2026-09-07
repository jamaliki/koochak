from collections import Counter
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

import scripts.submit_objective_decomposition_ca_ablation as launcher


def _parent(tmp_path: Path, monkeypatch) -> Path:
    config = {
        "model": {
            "node_dim": 768, "condition_dim": 256, "pair_dim": 64, "atom_dim": 128,
            "attention_heads": 12, "attention_head_dim": 64,
            "atom_heads": 4, "atom_head_dim": 32,
            "atom_encoder_depth": 1, "residue_encoder_depth": 4,
            "coarse_depth": 6, "residue_decoder_depth": 4, "atom_decoder_depth": 1,
            "atom_window_radius": 1, "pair_ffn_expansion": 4,
            "patchify_mode": "masked_pool", "coarse_pair_position": "before_attention",
            "coarse_pair_transition": True, "dropout": 0.2, "checkpoint_blocks": False,
        },
        "data": {
            "metadata_path": "/data/metadata.json", "min_length": 32, "max_length": 128,
            "mean_plddt_min": 80.0, "loop_length_max": 15,
            "loop_content_max": 0.4, "packing_density_min": 0.3,
            "batch_size": 256, "num_workers": 8, "length_buckets": [64, 96, 128],
            "prefetch_factor": 1, "shard_cache_size": None, "pin_memory": True,
            "persistent_workers": True, "seed": 42, "patch_capacities": [16, 24, 32],
        },
        "diffusion": {"p_mean": -1.2, "p_std": 1.5, "translation_std": 1.0},
        "loss": {
            "coordinate_weight": 1.0, "aatype_weight": 0.25,
            "smooth_lddt_weight": 1.0, "distogram_weight": 0.5,
        },
        "train": {
            "max_steps": 500_000, "log_every": 5, "ckpt_every": 50_000,
            "save_final": True, "grad_accum": 1, "grad_clip_norm": 1.0,
            "amp": "bf16", "ddp": False, "seed": 42, "keep_last_k": 12,
            "prefetch_batches": 2, "prefetch_threaded": True,
            "prefetch_pipeline": "two_stage", "self_conditioning_probability": 0.5,
            "compile": {"enabled": True, "mode": "default", "fullgraph": False, "dynamic": False},
            "require_compile": True, "require_fused": True,
        },
        "optimizer": {"name": "adam", "lr": 0.001},
        "logging": {"csv_path": "/parent/log.csv", "jsonl_path": "/parent/log.jsonl"},
        "sampling": {"num_steps": 200},
    }
    file = tmp_path / "parent.yaml"
    OmegaConf.save(OmegaConf.create(config), file)
    monkeypatch.setattr(launcher, "PARENT_CONFIG_SHA256", hashlib.sha256(file.read_bytes()).hexdigest())
    return file


def _manifest(task) -> dict:
    artifact = next(item for item in task.run.artifacts if item.path.endswith("launch.json"))
    return json.loads(artifact.content)


def test_builds_six_missing_decomposition_cells_two_ablations_and_exact_dag(
    tmp_path: Path, monkeypatch
) -> None:
    workflow, diffs = launcher.build_workflow(
        "a" * 40,
        parent_config=_parent(tmp_path, monkeypatch),
        output_root=tmp_path / "output",
    )
    counts = Counter(task.task_id.split("-", 1)[0] for task in workflow.tasks)

    assert len(launcher.DECOMPOSITION_CELLS) == 6
    assert len(launcher.ABLATION_CELLS) == 2
    assert len(workflow.tasks) == 36
    assert counts == Counter({
        "train": 8, "sample": 8, "esmfold": 8, "analysis": 8,
        "preflight": 2, "attest": 1, "aggregate": 1,
    })
    assert len(diffs) == 8
    assert {cell.cell_id for cell in launcher.DECOMPOSITION_CELLS}.isdisjoint(
        launcher.REUSED_ENDPOINTS
    )
    assert set(launcher.REUSED_ENDPOINTS) == {"lddt-m0g0c0", "lddt-m1g1c1"}
    assert all(
        task.to_scruffy_spec(
            request_id=workflow.request_id,
            workflow_id=workflow.workflow_id,
            project_id=workflow.project_id,
        )["recovery"] == launcher.RECOVERY
        for task in workflow.tasks
    )


def test_resolved_configs_change_only_declared_factors_and_preserve_io_invariants(
    tmp_path: Path, monkeypatch
) -> None:
    workflow, diffs = launcher.build_workflow(
        "b" * 40,
        parent_config=_parent(tmp_path, monkeypatch),
        output_root=tmp_path / "output",
    )
    trainers = {
        task.task_id.removeprefix("train-"): task
        for task in workflow.tasks
        if task.task_id.startswith("train-")
    }
    by_id = {item["cell_id"]: item for item in diffs}

    component = by_id["lddt-m0g0c1"]
    component_paths = {item["path"] for item in component["differences"]}
    assert component_paths == (
        set(launcher.STABILITY_VALUES)
        | {"loss.smooth_lddt_c_out_compensation"}
        | launcher.OPERATIONAL_PATHS
        | launcher.OUTPUT_PATHS
    )
    component_config = launcher._config_container(trainers["lddt-m0g0c1"].run)
    assert launcher._at(component_config, "loss.smooth_lddt_c_out_compensation") is True
    assert launcher._at(component_config, "loss.smooth_lddt_sigma_max", launcher._MISSING) is launcher._MISSING
    assert launcher._at(component_config, "loss.smooth_lddt_resolved_atom_only", launcher._MISSING) is launcher._MISSING

    ca_config = launcher._config_container(trainers["coordseq-ca"].run)
    assert launcher._at(ca_config, "model.atom_representation") == "ca"
    assert launcher._at(ca_config, "loss.smooth_lddt_weight") == 0.0
    assert launcher._at(ca_config, "loss.distogram_weight") == 0.0
    assert launcher._at(ca_config, "data.shard_cache_size") is None
    assert launcher._at(ca_config, "data.batch_size") == 256
    assert launcher._at(ca_config, "train.ddp") is False
    assert launcher._at(ca_config, "train.grad_accum") == 1
    assert launcher._at(ca_config, "optimizer.lr") == 0.001
    assert all(not item["unexpected_differences"] for item in diffs)


def test_only_novel_representation_cells_have_preflights_and_all_use_ack300(
    tmp_path: Path, monkeypatch
) -> None:
    workflow, _ = launcher.build_workflow(
        "c" * 40,
        parent_config=_parent(tmp_path, monkeypatch),
        output_root=tmp_path / "output",
    )
    preflight_ids = {
        task.task_id.removeprefix("preflight-")
        for task in workflow.tasks
        if task.task_id.startswith("preflight-")
    }
    assert preflight_ids == {"coordseq-atom14", "coordseq-ca"}

    for task in workflow.tasks:
        manifest = _manifest(task)
        encoded = json.dumps(manifest)
        assert "KOOCHAK_SCRUFFY_ARTIFACT_ACK_TIMEOUT_SECONDS" in encoded
        assert '300' in encoded
    for cell_id, trainer in (
        (task.task_id.removeprefix("train-"), task)
        for task in workflow.tasks
        if task.task_id.startswith("train-")
    ):
        manifest = _manifest(trainer)
        assert manifest["argv"][-2:] == ["--resume", "auto"]
        if cell_id.startswith("coordseq-"):
            assert trainer.wait_for[0]["task_id"] == f"preflight-{cell_id}"
        else:
            assert trainer.wait_for == ()


def test_sampling_waits_for_exact_50k_checkpoint(tmp_path: Path, monkeypatch) -> None:
    workflow, _ = launcher.build_workflow(
        "d" * 40,
        parent_config=_parent(tmp_path, monkeypatch),
        output_root=tmp_path / "output",
    )
    for task in workflow.tasks:
        if not task.task_id.startswith("sample-"):
            continue
        assert task.wait_for[0]["artifact_id"] == "checkpoint/step000050000.pt"
        assert "step000050000.pt" in " ".join(_manifest(task)["argv"])
