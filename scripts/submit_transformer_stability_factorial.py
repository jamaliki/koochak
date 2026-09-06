#!/usr/bin/env python3
"""Submit the L128 transformer-stability factorial and flat-baseline recovery."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    DeclaredOutput,
    PreparedRun,
    PreparedTask,
    PreparedWorkflow,
    load_environment_profile,
    prepare_run,
    submit_scruffy_workflow,
)


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
KOOCHAK_COMMIT = "eec841f0261c77c31d55100b1dd8af1e753903f5"
SCRUFFY_COMMIT = "0747640131470acd8ff30fe1b3be042139345224"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path(
    "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/"
    "scruffy-mcp-current/site"
)
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = REMOTE_CODE_ROOT / "hierarchical-kaveh-runs"
METADATA = Path(
    "/mnt/lustre/users/kiarash-eitgbi/atom14/afdb_all_parsed/"
    "parsed_np_shards_with_ss_3di/metadata_ca4_patch4.json"
)
PROGRES_DATA = REMOTE_CODE_ROOT / "progres-data" / "v1.1.0"
PROGRES_FILES = {
    "trained_model.pt": {
        "md5": "c490293eb8d0bb350e68a8229c6884da",
        "path": PROGRES_DATA / "trained_model.pt",
    },
    "cath40.pt": {
        "md5": "616510a3b21d45500b26dd5ae9a040d6",
        "path": PROGRES_DATA / "cath40.pt",
    },
}
PARENT_COMMIT = "97ce298cf0f5909ac0cbf50bdf94ab0481fbea8c"
PARENT_WORKFLOW = "hk-patch-coarse-factorial-500k-L128-followup-97ce298"
PARENT_RUN_ROOT = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-followup" / PARENT_COMMIT
GPU_PROFILE = REPO_ROOT / "environments" / "tokyo-factorial-gpu.yaml"
ESMFOLD_PROFILE = REPO_ROOT / "environments" / "tokyo-factorial-esmfold.yaml"
PROGRES_PROFILE = REPO_ROOT / "environments" / "tokyo-progres-cpu.yaml"
LENGTH = 128
MILESTONES = tuple(range(50_000, 500_001, 50_000))
SAMPLES_PER_LENGTH = 32
SAMPLE_SEED = 20260901
TRAIN_CHECKPOINT_INTERVAL = 10_000
TRAIN_KEEP_LAST_K = 60

TRAIN_RESOURCES = {
    "nodes": 1,
    "gpus_per_node": 1,
    "cpus_per_node": 14,
    "memory_gb_per_node": 240,
    "time_limit_seconds": 259_200,
}
SAMPLE_RESOURCES = {
    "nodes": 1,
    "gpus_per_node": 1,
    "cpus_per_node": 14,
    "memory_gb_per_node": 128,
    "time_limit_seconds": 21_600,
}
ESMFOLD_RESOURCES = {
    "nodes": 1,
    "gpus_per_node": 1,
    "cpus_per_node": 8,
    "memory_gb_per_node": 128,
    "time_limit_seconds": 43_200,
}
ANALYSIS_RESOURCES = {
    "nodes": 1,
    "gpus_per_node": 0,
    "cpus_per_node": 8,
    "memory_gb_per_node": 32,
    "time_limit_seconds": 14_400,
}
RECOVERY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
    "evacuation": {"signal": "USR1", "grace_seconds": 600},
}

ARCHITECTURES = (
    "flat_after_node_no_transition",
    "pool_before_attention_pair_transition",
)
STABILITY_SETTINGS = (
    ("depth", False, "depth-attn-no-sandwich"),
    ("full", True, "full-attn-sandwich"),
    ("depth", True, "depth-attn-sandwich"),
)
OUTPUT_DIFF_PATHS = {
    "train.out_dir",
    "logging.csv_path",
    "logging.jsonl_path",
}
OPERATIONAL_DIFF_PATHS = {"train.ckpt_every", "train.keep_last_k"}
FACTOR_DIFF_PATHS = {
    "model.attention_residual_scale",
    "model.sandwich_rmsnorm",
}

PARENT_CELLS = {
    architecture: PARENT_RUN_ROOT / "train" / "L128" / f"{architecture}-strict-sc0p5" / "config.yaml"
    for architecture in ARCHITECTURES
}
FLAT_SOURCE_CHECKPOINT = (
    PARENT_RUN_ROOT
    / "train/L128/flat_after_node_no_transition-strict-sc0p5/step000450000.pt"
)
FLAT_BASELINE_JOB = "job-3f732b074e68a4c49cab"


@dataclass(frozen=True)
class Cell:
    architecture: str
    attention_residual_scale: str
    sandwich_rmsnorm: bool
    setting_id: str

    @property
    def cell_id(self) -> str:
        return f"{self.architecture}-{self.setting_id}"


CELLS = tuple(
    Cell(architecture, attention_scale, sandwich, setting_id)
    for architecture in ARCHITECTURES
    for attention_scale, sandwich, setting_id in STABILITY_SETTINGS
)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _tag(step: int) -> str:
    return f"step{step:09d}"


def progres_data_spec(root: Path = PROGRES_DATA) -> dict[str, Any]:
    """Return the pinned Progres root and files, rejecting the old run-root path."""
    resolved_root = root.resolve()
    if resolved_root != PROGRES_DATA:
        raise ValueError(
            f"Progres data must use {PROGRES_DATA}, not {resolved_root}"
        )
    return {
        "root": str(resolved_root),
        "files": {
            name: {"path": str(spec["path"]), "md5": spec["md5"]}
            for name, spec in PROGRES_FILES.items()
        },
    }


def _parent_patches(
    parent: Path,
    *,
    attention_residual_scale: str | None = None,
    sandwich_rmsnorm: bool | None = None,
    run_dir: Path,
    stage: str,
) -> list[ConfigPatch]:
    patches: list[ConfigPatch] = []
    if attention_residual_scale is not None:
        patches.append(ConfigPatch("model.attention_residual_scale", attention_residual_scale))
    if sandwich_rmsnorm is not None:
        patches.append(ConfigPatch("model.sandwich_rmsnorm", sandwich_rmsnorm))
    patches.extend(
        [
            ConfigPatch("logging.csv_path", str(run_dir / f"{stage}.csv")),
            ConfigPatch("logging.jsonl_path", str(run_dir / f"{stage}.jsonl")),
        ]
    )
    return patches


def _trainer_patches(cell: Cell, run_dir: Path) -> list[ConfigPatch]:
    return [
        *_parent_patches(
            PARENT_CELLS[cell.architecture],
            attention_residual_scale=cell.attention_residual_scale,
            sandwich_rmsnorm=cell.sandwich_rmsnorm,
            run_dir=run_dir,
            stage="train",
        ),
        ConfigPatch("train.ckpt_every", TRAIN_CHECKPOINT_INTERVAL),
        ConfigPatch("train.keep_last_k", TRAIN_KEEP_LAST_K),
    ]


def _recovery_patches(run_dir: Path) -> list[ConfigPatch]:
    return [
        *_parent_patches(
            PARENT_CELLS["flat_after_node_no_transition"],
            run_dir=run_dir,
            stage="recovery",
        ),
        ConfigPatch("train.ckpt_every", TRAIN_CHECKPOINT_INTERVAL),
        ConfigPatch("train.keep_last_k", TRAIN_KEEP_LAST_K),
    ]


def _preflight_patches(cell: Cell, run_dir: Path) -> list[ConfigPatch]:
    return [
        *_trainer_patches(cell, run_dir),
        ConfigPatch("train.max_steps", 64),
        ConfigPatch("train.log_every", 1),
        ConfigPatch("train.save_final", False),
    ]


def _config_container(prepared: PreparedRun) -> dict[str, Any]:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    return OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, child_prefix))
        return flattened
    return {prefix: value}


def _resolved_diff(parent_config: Path, child: dict[str, Any], cell: Cell) -> dict[str, Any]:
    if not parent_config.is_file():
        raise FileNotFoundError(f"immutable parent config is missing: {parent_config}")
    parent = OmegaConf.to_container(OmegaConf.load(parent_config), resolve=True)
    parent_flat = _flatten(parent)
    child_flat = _flatten(child)
    differences = []
    for key in sorted(set(parent_flat) | set(child_flat)):
        if parent_flat.get(key) == child_flat.get(key):
            continue
        if key in FACTOR_DIFF_PATHS:
            classification = "scientific_factor"
        elif key in OPERATIONAL_DIFF_PATHS:
            classification = "explicit_operational"
        elif key in OUTPUT_DIFF_PATHS:
            classification = "run_identity_or_output"
        else:
            classification = "unexpected"
        differences.append(
            {
                "path": key,
                "parent_present": key in parent_flat,
                "child_present": key in child_flat,
                "parent": parent_flat.get(key),
                "child": child_flat.get(key),
                "classification": classification,
            }
        )
    expected = FACTOR_DIFF_PATHS | OPERATIONAL_DIFF_PATHS | OUTPUT_DIFF_PATHS
    observed = {item["path"] for item in differences}
    if observed != expected:
        raise AssertionError(
            f"unexpected resolved-config differences for {cell.cell_id}: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )
    if any(item["classification"] == "unexpected" for item in differences):
        raise AssertionError(f"unexpected config classification for {cell.cell_id}")
    expected_values = {
        "model.attention_residual_scale": cell.attention_residual_scale,
        "model.sandwich_rmsnorm": cell.sandwich_rmsnorm,
        "train.ckpt_every": TRAIN_CHECKPOINT_INTERVAL,
        "train.keep_last_k": TRAIN_KEEP_LAST_K,
    }
    for key, expected_value in expected_values.items():
        if child_flat.get(key) != expected_value:
            raise AssertionError(f"unexpected child value for {cell.cell_id}: {key}")
    return {
        "cell_id": cell.cell_id,
        "architecture": cell.architecture,
        "attention_residual_scale": cell.attention_residual_scale,
        "sandwich_rmsnorm": cell.sandwich_rmsnorm,
        "parent_config": str(parent_config),
        "parent_config_sha256": hashlib.sha256(parent_config.read_bytes()).hexdigest(),
        "differences": differences,
        "allowed_scientific_factor_paths": sorted(FACTOR_DIFF_PATHS),
        "allowed_operational_paths": sorted(OPERATIONAL_DIFF_PATHS),
        "allowed_output_paths": sorted(OUTPUT_DIFF_PATHS),
    }


def _recovery_diff(parent_config: Path, child: dict[str, Any]) -> dict[str, Any]:
    if not parent_config.is_file():
        raise FileNotFoundError(f"immutable recovery parent config is missing: {parent_config}")
    parent_flat = _flatten(OmegaConf.to_container(OmegaConf.load(parent_config), resolve=True))
    child_flat = _flatten(child)
    differences = [
        {
            "path": key,
            "parent_present": key in parent_flat,
            "child_present": key in child_flat,
            "parent": parent_flat.get(key),
            "child": child_flat.get(key),
            "classification": (
                "explicit_operational"
                if key in OPERATIONAL_DIFF_PATHS
                else "run_identity_or_output"
                if key in OUTPUT_DIFF_PATHS
                else "unexpected"
            ),
        }
        for key in sorted(set(parent_flat) | set(child_flat))
        if parent_flat.get(key) != child_flat.get(key)
    ]
    expected = OPERATIONAL_DIFF_PATHS | OUTPUT_DIFF_PATHS
    observed = {item["path"] for item in differences}
    if observed != expected or any(item["classification"] == "unexpected" for item in differences):
        raise AssertionError(
            f"unexpected recovery config differences: observed={sorted(observed)}, "
            f"expected={sorted(expected)}"
        )
    return {
        "cell_id": "flat_after_node_no_transition-strict-sc0p5-recovery-from450k",
        "parent_config": str(parent_config),
        "parent_config_sha256": hashlib.sha256(parent_config.read_bytes()).hexdigest(),
        "source_checkpoint": str(FLAT_SOURCE_CHECKPOINT),
        "source_step": 450_000,
        "differences": differences,
        "allowed_operational_paths": sorted(OPERATIONAL_DIFF_PATHS),
        "allowed_output_paths": sorted(OUTPUT_DIFF_PATHS),
    }


def _output(
    artifact_id: str,
    output_file: Path,
    *,
    stage: str,
    workflow: str,
    task: str,
    kind: str,
    expected_records: int | None = None,
) -> DeclaredOutput:
    return DeclaredOutput(
        artifact_id,
        str(output_file),
        kind=kind,
        stage=stage,
        expected_records=expected_records,
        provenance={
            "project_id": PROJECT_ID,
            "workflow_id": workflow,
            "task_id": task,
            "code_commit": _git("rev-parse", "HEAD"),
        },
        metadata={"stage": stage, "workflow": workflow},
    )


def _progres_attestation(
    *, workflow: str, output_root: Path, profile: Any
) -> tuple[PreparedTask, str]:
    task_id = "attest-progres-data"
    attestation_file = output_root / "progres-data" / "attestation.json"
    output = _output(
        "progres-data/attestation.json",
        attestation_file,
        stage="attestation",
        workflow=workflow,
        task=task_id,
        kind="file",
        expected_records=1,
    )
    run = _stage_run(
        stage="attestation",
        task=task_id,
        workflow=workflow,
        artifact=output,
        run_dir=attestation_file.parent.with_name(attestation_file.parent.name + ".managed"),
        profile=profile,
        command=[
            "{cwd}/scripts/attest_progres_data.py",
            "--data-dir",
            str(PROGRES_DATA),
            "--output",
            str(attestation_file),
        ],
    )
    return PreparedTask(task_id, run, ANALYSIS_RESOURCES, recovery=RECOVERY), output.artifact_id


def _profile(source: Path):
    loaded = load_environment_profile(source)
    variables = {
        key: value.replace(
            "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/"
            "scruffy-d60afabf-py310-cpython310-linux-x86_64/site",
            str(SCRUFFY_SITE),
        )
        for key, value in loaded.variables.items()
    }
    return replace(loaded, variables=variables)


def _stage_run(
    *,
    stage: str,
    task: str,
    workflow: str,
    artifact: DeclaredOutput,
    run_dir: Path,
    profile: Any,
    command: list[str],
    base_config: Path | None = None,
    patches: list[ConfigPatch] | None = None,
) -> PreparedRun:
    arguments = [
        "{cwd}/scripts/robust_factorial_stage.py",
        "--stage",
        stage,
        "--artifact-id",
        artifact.artifact_id,
        "--artifact-path",
        artifact.path,
        "--kind",
        artifact.kind,
        "--project",
        PROJECT_ID,
        "--workflow",
        workflow,
        "--task",
        task,
        "--code-commit",
        _git("rev-parse", "HEAD"),
    ]
    if artifact.expected_records is not None:
        arguments.extend(["--expected-records", str(artifact.expected_records)])
    arguments.extend(["--", *command])
    return prepare_run(
        name=f"hk-stability-{stage}-{task}",
        profile=profile,
        python_args=arguments,
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(run_dir),
        base_config=base_config,
        patches=patches or [],
        declared_outputs=[artifact],
    )


def _cell_train(
    cell: Cell, workflow: str, output_root: Path, profiles: Mapping[str, Any]
) -> tuple[PreparedRun, Path, list[ConfigPatch]]:
    train_dir = output_root / "train" / "L128" / cell.cell_id
    patches = _trainer_patches(cell, train_dir)
    run = prepare_run(
        name=f"hk-stability-train-{cell.cell_id}-{_git('rev-parse', 'HEAD')[:7]}",
        profile=profiles["gpu"],
        python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(train_dir),
        base_config=PARENT_CELLS[cell.architecture],
        patches=patches,
    )
    return run, train_dir, patches


def _recovery_train(
    workflow: str, output_root: Path, profiles: Mapping[str, Any]
) -> tuple[PreparedRun, Path, list[ConfigPatch]]:
    cell_id = "flat_after_node_no_transition-strict-sc0p5-recovery-from450k"
    train_dir = output_root / "recovery" / "train" / "L128" / cell_id
    patches = _recovery_patches(train_dir)
    run = prepare_run(
        name=f"hk-stability-recover-{cell_id}-{_git('rev-parse', 'HEAD')[:7]}",
        profile=profiles["gpu"],
        python_args=[
            "{cwd}/scripts/continue_training.py",
            "--config",
            "{config}",
            "--source-checkpoint",
            str(FLAT_SOURCE_CHECKPOINT),
            "--source-step",
            "450000",
            "--target-step",
            "500000",
        ],
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(train_dir),
        base_config=PARENT_CELLS["flat_after_node_no_transition"],
        patches=patches,
    )
    return run, train_dir, patches


def _build_cell_stages(
    *,
    cell: Cell,
    workflow: str,
    output_root: Path,
    profiles: Mapping[str, Any],
    train_id: str,
    train: PreparedRun,
    train_dir: Path,
    train_patches: list[ConfigPatch],
    progres_attestation_artifact: str,
    tasks: list[PreparedTask],
) -> None:
    preflight_id = f"preflight-{cell.cell_id}"
    preflight_dir = output_root / "preflight" / cell.cell_id
    preflight_report = preflight_dir / "report.json"
    preflight_output = _output(
        f"preflight/{cell.cell_id}/report.json",
        preflight_report,
        stage="preflight",
        workflow=workflow,
        task=preflight_id,
        kind="file",
        expected_records=1,
    )
    preflight_run = _stage_run(
        stage="preflight",
        task=preflight_id,
        workflow=workflow,
        artifact=preflight_output,
        run_dir=preflight_dir.with_name(preflight_dir.name + ".managed"),
        profile=profiles["gpu"],
        base_config=PARENT_CELLS[cell.architecture],
        patches=_preflight_patches(cell, preflight_dir.with_name(preflight_dir.name + ".managed")),
        command=[
            "{cwd}/scripts/stability_preflight.py",
            "--config",
            "{config}",
            "--output",
            str(preflight_report),
            "--cell-id",
            cell.cell_id,
            "--expected-workers",
            "8",
            "--warmup-steps",
            "16",
            "--minimum-timed-rows",
            "16",
        ],
    )
    tasks.append(PreparedTask(preflight_id, preflight_run, TRAIN_RESOURCES, recovery=RECOVERY))
    tasks.append(
        PreparedTask(
            train_id,
            train,
            TRAIN_RESOURCES,
            wait_for=({"kind": "artifact", "task_id": preflight_id, "artifact_id": preflight_output.artifact_id},),
            recovery=RECOVERY,
        )
    )

    for step in MILESTONES:
        tag = _tag(step)
        checkpoint = train_dir / f"{tag}.pt"
        checkpoint_id = f"checkpoint/{tag}.pt"
        sample_id = f"sample-{cell.cell_id}-{tag}"
        sample_dir = output_root / "samples" / tag / cell.cell_id
        sample_output = _output(
            f"samples/{tag}/{cell.cell_id}",
            sample_dir,
            stage="sample",
            workflow=workflow,
            task=sample_id,
            kind="directory",
            expected_records=SAMPLES_PER_LENGTH,
        )
        sample_run = _stage_run(
            stage="sample",
            task=sample_id,
            workflow=workflow,
            artifact=sample_output,
            run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
            profile=profiles["gpu"],
            base_config=PARENT_CELLS[cell.architecture],
            patches=_parent_patches(
                PARENT_CELLS[cell.architecture],
                attention_residual_scale=cell.attention_residual_scale,
                sandwich_rmsnorm=cell.sandwich_rmsnorm,
                run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
                stage="sample",
            ),
            command=[
                "{cwd}/scripts/sample_short128_milestone.py",
                "--config",
                "{config}",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(sample_dir),
                "--lengths",
                "128",
                "--samples-per-length",
                str(SAMPLES_PER_LENGTH),
                "--batch-size",
                "8",
                "--seed",
                str(SAMPLE_SEED),
                "--precision",
                "bf16",
                "--compile",
            ],
        )
        tasks.append(
            PreparedTask(
                sample_id,
                sample_run,
                SAMPLE_RESOURCES,
                wait_for=(
                    {"kind": "artifact", "task_id": preflight_id, "artifact_id": preflight_output.artifact_id},
                    {"kind": "artifact", "task_id": train_id, "artifact_id": checkpoint_id},
                ),
                recovery=RECOVERY,
            )
        )

        esmfold_id = f"esmfold-{cell.cell_id}-{tag}"
        esmfold_dir = output_root / "esmfold" / tag / cell.cell_id / "L0128"
        esmfold_output = _output(
            f"esmfold/{tag}/{cell.cell_id}/L0128",
            esmfold_dir,
            stage="esmfold",
            workflow=workflow,
            task=esmfold_id,
            kind="directory",
            expected_records=SAMPLES_PER_LENGTH,
        )
        esmfold_run = _stage_run(
            stage="esmfold",
            task=esmfold_id,
            workflow=workflow,
            artifact=esmfold_output,
            run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"),
            profile=profiles["esmfold"],
            command=[
                "{cwd}/scripts/run_esmfold_designability_shard.py",
                "--sample-dir",
                str(sample_dir / "L0128"),
                "--output-dir",
                str(esmfold_dir),
                "--variant",
                cell.cell_id,
                "--step",
                str(step),
                "--length",
                "128",
                "--expected-count",
                str(SAMPLES_PER_LENGTH),
                "--wrapper",
                "{cwd}/scripts/esmfold_predict_container",
                "--chunk-size",
                "8",
                "--bf16",
            ],
        )
        tasks.append(
            PreparedTask(
                esmfold_id,
                esmfold_run,
                ESMFOLD_RESOURCES,
                wait_for=(
                    {"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},
                ),
                recovery=RECOVERY,
            )
        )

        analysis_id = f"analysis-{cell.cell_id}-{tag}"
        analysis_file = output_root / "analysis" / tag / cell.cell_id / "progres_diversity.json"
        analysis_output = _output(
            f"analysis/{tag}/{cell.cell_id}/progres_diversity.json",
            analysis_file,
            stage="analysis",
            workflow=workflow,
            task=analysis_id,
            kind="file",
            expected_records=1,
        )
        analysis_run = _stage_run(
            stage="analysis",
            task=analysis_id,
            workflow=workflow,
            artifact=analysis_output,
            run_dir=analysis_file.parent.with_name(analysis_file.parent.name + ".managed"),
            profile=profiles["progres"],
            command=[
                "{cwd}/scripts/analyze_progres_diversity.py",
                "--sample-dir",
                str(sample_dir / "L0128"),
                "--esmfold-dir",
                str(esmfold_dir),
                "--data-dir",
                str(PROGRES_DATA),
                "--expected-count",
                str(SAMPLES_PER_LENGTH),
                "--panel",
                cell.cell_id,
                "--step",
                str(step),
                "--output",
                str(analysis_file),
            ],
        )
        tasks.append(
            PreparedTask(
                analysis_id,
                analysis_run,
                ANALYSIS_RESOURCES,
                wait_for=(
                    {"kind": "artifact", "task_id": esmfold_id, "artifact_id": esmfold_output.artifact_id},
                    {"kind": "artifact", "task_id": "attest-progres-data", "artifact_id": progres_attestation_artifact},
                ),
                recovery=RECOVERY,
            )
        )


def _build_recovery_stages(
    *,
    workflow: str,
    output_root: Path,
    profiles: Mapping[str, Any],
    recovery_id: str,
    recovery_dir: Path,
    progres_attestation_artifact: str,
    tasks: list[PreparedTask],
) -> None:
    step = 500_000
    tag = _tag(step)
    checkpoint = recovery_dir / f"{tag}.pt"
    checkpoint_id = f"checkpoint/{tag}.pt"
    sample_id = f"sample-recovery-flat-strict-sc0p5-{tag}"
    sample_dir = output_root / "recovery" / "samples" / tag / "flat-strict-sc0p5"
    sample_output = _output(
        f"recovery/samples/{tag}/flat-strict-sc0p5",
        sample_dir,
        stage="sample",
        workflow=workflow,
        task=sample_id,
        kind="directory",
        expected_records=SAMPLES_PER_LENGTH,
    )
    sample_run = _stage_run(
        stage="sample",
        task=sample_id,
        workflow=workflow,
        artifact=sample_output,
        run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
        profile=profiles["gpu"],
        base_config=PARENT_CELLS["flat_after_node_no_transition"],
        patches=_parent_patches(
            PARENT_CELLS["flat_after_node_no_transition"],
            run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
            stage="sample",
        ),
        command=[
            "{cwd}/scripts/sample_short128_milestone.py",
            "--config",
            "{config}",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(sample_dir),
            "--lengths",
            "128",
            "--samples-per-length",
            str(SAMPLES_PER_LENGTH),
            "--batch-size",
            "8",
            "--seed",
            str(SAMPLE_SEED),
            "--precision",
            "bf16",
            "--compile",
        ],
    )
    tasks.append(
        PreparedTask(
            sample_id,
            sample_run,
            SAMPLE_RESOURCES,
            wait_for=(
                {"kind": "artifact", "task_id": recovery_id, "artifact_id": checkpoint_id},
            ),
            recovery=RECOVERY,
        )
    )
    esmfold_id = f"esmfold-recovery-flat-strict-sc0p5-{tag}"
    esmfold_dir = output_root / "recovery" / "esmfold" / tag / "flat-strict-sc0p5" / "L0128"
    esmfold_output = _output(
        f"recovery/esmfold/{tag}/flat-strict-sc0p5/L0128",
        esmfold_dir,
        stage="esmfold",
        workflow=workflow,
        task=esmfold_id,
        kind="directory",
        expected_records=SAMPLES_PER_LENGTH,
    )
    esmfold_run = _stage_run(
        stage="esmfold",
        task=esmfold_id,
        workflow=workflow,
        artifact=esmfold_output,
        run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"),
        profile=profiles["esmfold"],
        command=[
            "{cwd}/scripts/run_esmfold_designability_shard.py",
            "--sample-dir",
            str(sample_dir / "L0128"),
            "--output-dir",
            str(esmfold_dir),
            "--variant",
            "flat-strict-sc0p5-recovery",
            "--step",
            str(step),
            "--length",
            "128",
            "--expected-count",
            str(SAMPLES_PER_LENGTH),
            "--wrapper",
            "{cwd}/scripts/esmfold_predict_container",
            "--chunk-size",
            "8",
            "--bf16",
        ],
    )
    tasks.append(
        PreparedTask(
            esmfold_id,
            esmfold_run,
            ESMFOLD_RESOURCES,
            wait_for=(
                {"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},
            ),
            recovery=RECOVERY,
        )
    )
    analysis_id = f"analysis-recovery-flat-strict-sc0p5-{tag}"
    analysis_file = output_root / "recovery" / "analysis" / tag / "flat-strict-sc0p5" / "progres_diversity.json"
    analysis_output = _output(
        f"recovery/analysis/{tag}/flat-strict-sc0p5/progres_diversity.json",
        analysis_file,
        stage="analysis",
        workflow=workflow,
        task=analysis_id,
        kind="file",
        expected_records=1,
    )
    analysis_run = _stage_run(
        stage="analysis",
        task=analysis_id,
        workflow=workflow,
        artifact=analysis_output,
        run_dir=analysis_file.parent.with_name(analysis_file.parent.name + ".managed"),
        profile=profiles["progres"],
        command=[
            "{cwd}/scripts/analyze_progres_diversity.py",
            "--sample-dir",
            str(sample_dir / "L0128"),
            "--esmfold-dir",
            str(esmfold_dir),
            "--data-dir",
            str(PROGRES_DATA),
            "--expected-count",
            str(SAMPLES_PER_LENGTH),
            "--panel",
            "flat-strict-sc0p5-recovery",
            "--step",
            str(step),
            "--output",
            str(analysis_file),
        ],
    )
    tasks.append(
        PreparedTask(
            analysis_id,
            analysis_run,
            ANALYSIS_RESOURCES,
            wait_for=(
                {"kind": "artifact", "task_id": esmfold_id, "artifact_id": esmfold_output.artifact_id},
                {"kind": "artifact", "task_id": "attest-progres-data", "artifact_id": progres_attestation_artifact},
            ),
            recovery=RECOVERY,
        )
    )


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, dict[str, Any]]:
    short = code_commit[:7]
    workflow = f"hk-transformer-stability-factorial-500k-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "transformer-stability-factorial-500k-L128" / code_commit
    profiles = {
        "gpu": _profile(GPU_PROFILE),
        "esmfold": _profile(ESMFOLD_PROFILE),
        "progres": _profile(PROGRES_PROFILE),
    }
    tasks: list[PreparedTask] = []
    diffs: list[dict[str, Any]] = []
    trainer_task_ids: dict[str, str] = {}
    progres_attestation, progres_attestation_artifact = _progres_attestation(
        workflow=workflow, output_root=output_root, profile=profiles["progres"]
    )
    tasks.append(progres_attestation)
    for cell in CELLS:
        train, train_dir, train_patches = _cell_train(cell, workflow, output_root, profiles)
        train_id = f"train-{cell.cell_id}"
        trainer_task_ids[cell.cell_id] = train_id
        diffs.append(_resolved_diff(PARENT_CELLS[cell.architecture], _config_container(train), cell))
        _build_cell_stages(
            cell=cell,
            workflow=workflow,
            output_root=output_root,
            profiles=profiles,
            train_id=train_id,
            train=train,
            train_dir=train_dir,
            train_patches=train_patches,
            progres_attestation_artifact=progres_attestation_artifact,
            tasks=tasks,
        )

    recovery_run, recovery_dir, _recovery_patch_values = _recovery_train(workflow, output_root, profiles)
    recovery_id = "recover-flat-strict-sc0p5-from450k"
    recovery_diff = _recovery_diff(
        PARENT_CELLS["flat_after_node_no_transition"], _config_container(recovery_run)
    )
    tasks.append(PreparedTask(recovery_id, recovery_run, TRAIN_RESOURCES, recovery=RECOVERY))
    _build_recovery_stages(
        workflow=workflow,
        output_root=output_root,
        profiles=profiles,
        recovery_id=recovery_id,
        recovery_dir=recovery_dir,
        progres_attestation_artifact=progres_attestation_artifact,
        tasks=tasks,
    )
    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )
    description = {
        "workflow_id": workflow,
        "request_id": prepared.request_id,
        "project_id": PROJECT_ID,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_controller_release": SCRUFFY_COMMIT,
        "parent_workflow": PARENT_WORKFLOW,
        "progres_data": progres_data_spec(),
        "progres_data_attestation": {
            "task_id": "attest-progres-data",
            "artifact_id": progres_attestation_artifact,
            "method": "single CPU-side chunked MD5 verification before all Progres analyses",
        },
        "parent_configs": {key: str(value) for key, value in PARENT_CELLS.items()},
        "reused_baseline_cells": [
            {
                "cell_id": f"{architecture}-strict-sc0p5",
                "architecture": architecture,
                "parent_config": str(PARENT_CELLS[architecture]),
                "baseline_action": "reuse_existing_no_duplicate",
            }
            for architecture in ARCHITECTURES
        ],
        "flat_baseline_recovery": {
            "task_id": "recover-flat-strict-sc0p5-from450k",
            "failed_source_job_id": FLAT_BASELINE_JOB,
            "source_checkpoint": str(FLAT_SOURCE_CHECKPOINT),
            "source_step": 450_000,
            "target_step": 500_000,
            "action": "resume_verified_source_checkpoint_with_own_auto_resume_only",
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "architecture": cell.architecture,
                "attention_residual_scale": cell.attention_residual_scale,
                "sandwich_rmsnorm": cell.sandwich_rmsnorm,
                "from_scratch": True,
                "trainer_task_id": trainer_task_ids[cell.cell_id],
            }
            for cell in CELLS
        ],
        "milestones": list(MILESTONES),
        "samples_per_length": SAMPLES_PER_LENGTH,
        "checkpoint_interval_steps": TRAIN_CHECKPOINT_INTERVAL,
        "checkpoint_retention": TRAIN_KEEP_LAST_K,
        "primary_endpoint": "generated designable IDs clustered by complete-linkage Progres at similarity >= 0.8",
        "secondary_endpoint": "ESMFold-refolded designable IDs clustered by complete-linkage Progres",
        "task_count": len(prepared.tasks),
        "task_counts": {
            "trainers": sum(task.task_id.startswith("train-") for task in prepared.tasks),
            "preflights": sum(task.task_id.startswith("preflight-") for task in prepared.tasks),
            "samples": sum(task.task_id.startswith("sample-") for task in prepared.tasks),
            "esmfold": sum(task.task_id.startswith("esmfold-") for task in prepared.tasks),
            "analysis": sum(task.task_id.startswith("analysis-") for task in prepared.tasks),
            "attestation": sum(task.task_id == "attest-progres-data" for task in prepared.tasks),
            "recovery": sum(task.task_id == "recover-flat-strict-sc0p5-from450k" for task in prepared.tasks),
        },
        "resolved_config_diffs": diffs,
        "recovery_config_diff": recovery_diff,
        "norm_semantics": {
            "attention_residual_scale": "full=1.0; depth=1/sqrt(2*(residue_encoder_depth+coarse_depth+residue_decoder_depth))",
            "sandwich_rmsnorm": "non-affine RMS normalization of the masked residual stream after each attention and FFN addition; raw updates are never normalized directly",
            "diagnostics": [
                "raw branch update RMS",
                "post-add stream RMS",
                "post-norm stream RMS",
            ],
        },
        "cache_contract": {
            "shard_cache_size": None,
            "ownership": "disjoint rank/worker physical-shard ownership with resident preload",
            "preflight_gate": {
                "workers": 8,
                "memory_gb": 240,
                "warmed_p50_seconds_max": 0.75,
                "warmed_p90_seconds_max": 1.5,
                "finite_loss_required": True,
                "post-warmup_cache_misses": 0,
            },
        },
    }
    return prepared, {"description": description, "output_root": output_root}


def _validate_source_manifest() -> None:
    source = FLAT_SOURCE_CHECKPOINT
    manifest = Path(f"{source}.ready.json")
    if source.is_symlink() or not source.is_file() or manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(f"verified recovery source or publication is unavailable: {source}")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        "v": 1,
        "artifact_id": "checkpoint/step000450000.pt",
        "path": str(source.absolute()),
        "manifest_path": str(manifest.absolute()),
        "size_bytes": source.stat().st_size,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"recovery source publication mismatch for {key}: {source}")
    if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
        raise RuntimeError(f"recovery source publication has no SHA-256: {manifest}")


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    koochak_root = (REPO_ROOT / "external" / "koochak").resolve()
    if _git("rev-parse", "HEAD", cwd=koochak_root) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected_checkout = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected_checkout:
        raise RuntimeError(f"run from the independent checkout {expected_checkout}")
    required = [SCRUFFY_ROOT, SCRUFFY_SITE, METADATA, PROGRES_DATA]
    missing = [str(item) for item in required if not item.exists()]
    if PROGRES_DATA.resolve() != REMOTE_CODE_ROOT / "progres-data" / "v1.1.0":
        raise RuntimeError(f"unexpected Progres root: {PROGRES_DATA}")
    for name, spec in PROGRES_FILES.items():
        file = spec["path"]
        if file.is_symlink() or not file.is_file():
            missing.append(str(file))
    missing.extend(str(config) for config in PARENT_CELLS.values() if not config.is_file())
    if missing:
        raise RuntimeError(f"required launch locations are missing: {missing}")
    _validate_source_manifest()


def _fresh_output(output_root: Path, train_dirs: list[Path]) -> None:
    if (output_root / "resolved_config_diffs.json").exists():
        raise RuntimeError(f"workflow output already has an attestation: {output_root}")
    for train_dir in train_dirs:
        if any(train_dir.glob("step*.pt")) or (train_dir / "latest.pt").exists():
            raise RuntimeError(f"refusing to duplicate a non-empty trainer output: {train_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _validate_online(code_commit)
    workflow, bundle = build_workflow(code_commit)
    description = bundle["description"]
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return
    output_root = bundle["output_root"]
    train_dirs = [
        output_root / "train" / "L128" / cell.cell_id for cell in CELLS
    ] + [
        output_root
        / "recovery"
        / "train"
        / "L128"
        / "flat_after_node_no_transition-strict-sc0p5-recovery-from450k"
    ]
    _fresh_output(output_root, train_dirs)
    output_root.mkdir(parents=True, exist_ok=False)
    attestation_path = output_root / "resolved_config_diffs.json"
    attestation_path.write_text(
        json.dumps(description, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(
            f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}"
        )
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(
        json.dumps(
            {
                "workflow": description,
                "attestation_path": str(attestation_path),
                "submission": submission,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
