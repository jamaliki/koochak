#!/usr/bin/env python3
"""Prepare or submit the 16-cell 50k Atom14 causal intervention factorial."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
    "scruffy-0747640131470acd8ff30fe1b3be042139345224-py310-cpython310/site"
)
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = REMOTE_CODE_ROOT / "hierarchical-kaveh-runs"
PARENT_COMMIT = "97ce298cf0f5909ac0cbf50bdf94ab0481fbea8c"
PARENT_WORKFLOW = "hk-patch-coarse-factorial-500k-L128-followup-97ce298"
PARENT_RUN_ROOT = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-followup" / PARENT_COMMIT
METADATA = Path(
    "/mnt/lustre/users/kiarash-eitgbi/atom14/afdb_all_parsed/"
    "parsed_np_shards_with_ss_3di/metadata_ca4_patch4.json"
)
PROGRES_DATA = REMOTE_CODE_ROOT / "progres-data" / "v1.1.0"
GPU_PROFILE = REPO_ROOT / "environments/tokyo-factorial-gpu.yaml"
CPU_PROFILE = REPO_ROOT / "environments/tokyo-factorial-cpu.yaml"
ESMFOLD_PROFILE = REPO_ROOT / "environments/tokyo-factorial-esmfold.yaml"
PROGRES_PROFILE = REPO_ROOT / "environments/tokyo-progres-cpu.yaml"

ARCHITECTURES = (
    "flat_after_node_no_transition",
    "pool_before_attention_pair_transition",
)
ARCHITECTURE_KEYS = {
    "flat_after_node_no_transition": {
        "model.patchify_mode": "flat_linear",
        "model.coarse_pair_position": "after_node",
        "model.coarse_pair_transition": False,
    },
    "pool_before_attention_pair_transition": {
        "model.patchify_mode": "masked_pool",
        "model.coarse_pair_position": "before_attention",
        "model.coarse_pair_transition": True,
    },
}
PARENT_CELLS: dict[str, dict[str, Any]] = {
    "flat_after_node_no_transition": {
        "config": PARENT_RUN_ROOT / "train/L128/flat_after_node_no_transition-strict-sc0p5/config.yaml",
        "sha256": "fac4a61656752c4a00e0124b46dfd6a12fdeb66525298c91c48df0a4618433ac",
    },
    "pool_before_attention_pair_transition": {
        "config": PARENT_RUN_ROOT / "train/L128/pool_before_attention_pair_transition-strict-sc0p5/config.yaml",
        "sha256": "f33a72cfa082957ed5004c4e06d00ca1662fd39e59357926431f779015ff4196",
    },
}

MAX_STEPS = 50_000
CHECKPOINT_STEP = 50_000
TRAIN_CHECKPOINT_INTERVAL = 10_000
LENGTH = 128
LENGTH_BUCKETS = [64, 96, 128]
PATCH_CAPACITIES = [16, 24, 32]
SAMPLES_PER_LENGTH = 32
SAMPLE_BATCH_SIZE = 8
SAMPLE_SEED = 20260901
SIGMA_PROBE_SEED = 20260906
SIGMA_PROBE_SAMPLES = 4096
PREFLIGHT_STEPS = 64

OBJECTIVE_VALUES = {
    "loss.smooth_lddt_sigma_max": 3.0,
    "loss.smooth_lddt_resolved_atom_only": True,
    "loss.smooth_lddt_c_out_compensation": True,
}
RESIDUAL_VALUES = {
    "model.atom_ffn_residual_scale": "depth",
    "model.atom_to_residue_mean_rmsnorm": True,
    "model.atom_to_residue_residual_scale": "depth",
}
TRANSPORT_VALUES = {"model.atom_to_residue_transport": "backbone_first"}
OBJECTIVE_PATHS = set(OBJECTIVE_VALUES)
RESIDUAL_PATHS = set(RESIDUAL_VALUES)
TRANSPORT_PATHS = set(TRANSPORT_VALUES)
FACTOR_PATHS = OBJECTIVE_PATHS | RESIDUAL_PATHS | TRANSPORT_PATHS
OPERATIONAL_PATHS = {"train.max_steps", "train.ckpt_every"}
OUTPUT_PATHS = {"train.out_dir", "logging.csv_path", "logging.jsonl_path"}
ALLOWED_DIFF_PATHS = FACTOR_PATHS | OPERATIONAL_PATHS | OUTPUT_PATHS
_MISSING = object()

TRAIN_RESOURCES = {
    "nodes": 1, "gpus_per_node": 1, "cpus_per_node": 14,
    "memory_gb_per_node": 240, "time_limit_seconds": 259_200,
}
PREFLIGHT_RESOURCES = {
    "nodes": 1, "gpus_per_node": 1, "cpus_per_node": 14,
    "memory_gb_per_node": 240, "time_limit_seconds": 21_600,
}
RESOURCES = {
    "sample": {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 14, "memory_gb_per_node": 128, "time_limit_seconds": 21_600},
    "esmfold": {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 8, "memory_gb_per_node": 128, "time_limit_seconds": 43_200},
    "cpu": {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 8, "memory_gb_per_node": 32, "time_limit_seconds": 14_400},
}
RECOVERY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
    "evacuation": {"signal": "USR1", "grace_seconds": 600},
}


@dataclass(frozen=True, slots=True)
class Cell:
    architecture: str
    objective: int
    residual: int
    transport: int

    @property
    def cell_id(self) -> str:
        return f"{self.architecture}-o{self.objective}r{self.residual}t{self.transport}"


CELLS = tuple(
    Cell(architecture, objective, residual, transport)
    for architecture in ARCHITECTURES
    for objective in (0, 1)
    for residual in (0, 1)
    for transport in (0, 1)
)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _replace_all(value: str, replacements: Mapping[str, str]) -> str:
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _load_profile(source: Path):
    profile = load_environment_profile(source)
    replacements = {
        "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site": str(SCRUFFY_SITE),
        "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-d60afabf-py310-cpython310-linux-x86_64/site": str(SCRUFFY_SITE),
    }
    variables = {key: _replace_all(value, replacements) for key, value in profile.variables.items()}
    return replace(profile, variables=variables, profile_id=f"{profile.profile_id}-py310")


def _parent_config(cell: Cell, parent_cells: Mapping[str, Mapping[str, Any]]) -> Path:
    try:
        return Path(parent_cells[cell.architecture]["config"])
    except KeyError as error:
        raise KeyError(f"no immutable parent config for {cell.architecture}") from error


def _factor_patches(cell: Cell) -> list[ConfigPatch]:
    groups = (
        (cell.objective, OBJECTIVE_VALUES),
        (cell.residual, RESIDUAL_VALUES),
        (cell.transport, TRANSPORT_VALUES),
    )
    return [ConfigPatch(key, value) for enabled, values in groups if enabled for key, value in values.items()]


def _logging_patches(run_dir: Path) -> list[ConfigPatch]:
    return [
        ConfigPatch("logging.csv_path", str(run_dir / "metrics.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "metrics.jsonl")),
    ]


def _trainer_patches(cell: Cell, run_dir: Path) -> list[ConfigPatch]:
    return [
        *_factor_patches(cell),
        ConfigPatch("train.max_steps", MAX_STEPS),
        ConfigPatch("train.ckpt_every", TRAIN_CHECKPOINT_INTERVAL),
        *_logging_patches(run_dir),
    ]


def _evaluation_patches(cell: Cell, run_dir: Path) -> list[ConfigPatch]:
    return [*_factor_patches(cell), *_logging_patches(run_dir)]


def _preflight_patches(cell: Cell, run_dir: Path) -> list[ConfigPatch]:
    return [
        *_evaluation_patches(cell, run_dir),
        ConfigPatch("train.max_steps", PREFLIGHT_STEPS),
        ConfigPatch("train.log_every", 1),
        ConfigPatch("train.ckpt_every", TRAIN_CHECKPOINT_INTERVAL),
        ConfigPatch("train.save_final", False),
    ]


def _config_container(prepared: PreparedRun) -> dict[str, Any]:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    value = OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config must be a mapping")
    return value


def _config_artifact(prepared: PreparedRun):
    return next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def _at(config: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _assert_invariants(config: Mapping[str, Any], cell: Cell) -> None:
    expected = {
        **ARCHITECTURE_KEYS[cell.architecture],
        "data.metadata_path": str(METADATA), "data.min_length": 32, "data.max_length": 128,
        "data.mean_plddt_min": 80.0, "data.loop_length_max": 15,
        "data.loop_content_max": 0.4, "data.packing_density_min": 0.3,
        "data.batch_size": 256, "data.num_workers": 8, "data.prefetch_factor": 1,
        "data.length_buckets": LENGTH_BUCKETS, "data.patch_capacities": PATCH_CAPACITIES,
        "data.pin_memory": True, "data.persistent_workers": True, "data.shard_cache_size": None,
        "train.ddp": False, "train.grad_accum": 1, "train.amp": "bf16",
        "train.self_conditioning_probability": 0.5, "train.compile.enabled": True,
        "train.require_compile": True, "train.require_fused": True,
        "train.prefetch_batches": 2, "train.prefetch_threaded": True,
        "train.prefetch_pipeline": "two_stage", "optimizer.name": "adam",
        "optimizer.lr": 0.001, "optimizer.weight_decay": 0.0,
        "optimizer.betas": [0.9, 0.999], "optimizer.eps": 1.0e-8,
        "train.max_steps": MAX_STEPS, "train.ckpt_every": TRAIN_CHECKPOINT_INTERVAL,
        "train.keep_last_k": 12,
    }
    for dotted, wanted in expected.items():
        observed = _at(config, dotted)
        if observed is _MISSING:
            raise AssertionError(f"required invariant is absent: {dotted}")
        if observed != wanted:
            raise AssertionError(f"{dotted}: expected {wanted!r}, got {observed!r}")
    for dotted, wanted in {
        "model.sigma_data": 16.0, "loss.aatype_sigma_max": 0.5,
        "loss.aatype_sigma_ramp_max": None, "model.progres_conditioning": False,
    }.items():
        observed = _at(config, dotted)
        if observed is not _MISSING and observed != wanted:
            raise AssertionError(f"effective default mismatch for {dotted}: {observed!r}")


def _assert_rendered_config(prepared: PreparedRun, base_config: Path, patches: Sequence[ConfigPatch]) -> None:
    expected = OmegaConf.load(base_config)
    OmegaConf.update(expected, "train.out_dir", prepared.run_dir, force_add=True)
    for patch in patches:
        OmegaConf.update(expected, patch.path, patch.value, merge=patch.merge, force_add=True)
    if _config_container(prepared) != OmegaConf.to_container(expected, resolve=True):
        raise AssertionError(f"rendered config differs from patches: {prepared.name}")


def _classification(dotted: str) -> str:
    if dotted in FACTOR_PATHS:
        return "scientific_factor"
    if dotted in OPERATIONAL_PATHS:
        return "explicit_invariant"
    return "run_identity_output"


def _resolved_diff(parent_config: Path, child: Mapping[str, Any], cell: Cell, *, child_sha256: str | None = None, expected_parent_sha256: str | None = None) -> dict[str, Any]:
    if not parent_config.is_file():
        raise FileNotFoundError(f"immutable parent config is missing: {parent_config}")
    parent_sha256 = hashlib.sha256(parent_config.read_bytes()).hexdigest()
    if expected_parent_sha256 is not None and parent_sha256 != expected_parent_sha256:
        raise AssertionError(f"parent config hash mismatch: expected {expected_parent_sha256}, got {parent_sha256}")
    parent = OmegaConf.to_container(OmegaConf.load(parent_config), resolve=True)
    parent_flat, child_flat = _flatten(parent), _flatten(child)
    differences = []
    for dotted in sorted(set(parent_flat) | set(child_flat)):
        parent_value, child_value = parent_flat.get(dotted, _MISSING), child_flat.get(dotted, _MISSING)
        if parent_value == child_value:
            continue
        differences.append({
            "path": dotted, "parent_present": parent_value is not _MISSING,
            "child_present": child_value is not _MISSING,
            "parent": None if parent_value is _MISSING else parent_value,
            "child": None if child_value is _MISSING else child_value,
            "classification": _classification(dotted),
        })
    observed = {item["path"] for item in differences}
    unexpected = sorted(observed - ALLOWED_DIFF_PATHS)
    if unexpected:
        raise AssertionError(f"unexpected resolved-config differences for {cell.cell_id}: {unexpected}")

    enabled_values: dict[str, Any] = {}
    if cell.objective:
        enabled_values.update(OBJECTIVE_VALUES)
    if cell.residual:
        enabled_values.update(RESIDUAL_VALUES)
    if cell.transport:
        enabled_values.update(TRANSPORT_VALUES)
    for dotted in FACTOR_PATHS:
        parent_value, child_value = parent_flat.get(dotted, _MISSING), child_flat.get(dotted, _MISSING)
        if dotted in enabled_values:
            if child_value != enabled_values[dotted]:
                raise AssertionError(f"factor-on value mismatch for {cell.cell_id}: {dotted}")
        elif child_value != parent_value:
            raise AssertionError(f"factor-off cell changed or materialized {dotted}: {cell.cell_id}")
    for dotted, wanted in {"train.max_steps": MAX_STEPS, "train.ckpt_every": TRAIN_CHECKPOINT_INTERVAL}.items():
        if child_flat.get(dotted, _MISSING) != wanted:
            raise AssertionError(f"{dotted} mismatch for {cell.cell_id}")
    _assert_invariants(child, cell)
    return {
        "schema": "hierarchical-kaveh.atom14-causal-intervention-config-diff.v1",
        "cell_id": cell.cell_id, "architecture": cell.architecture,
        "factors": {"objective": cell.objective, "residual": cell.residual, "transport": cell.transport},
        "parent_config": str(parent_config), "parent_config_sha256": parent_sha256,
        "expected_parent_config_sha256": expected_parent_sha256,
        "child_config_sha256": child_sha256, "allowed_paths": sorted(ALLOWED_DIFF_PATHS),
        "differences": differences, "unexpected_differences": unexpected,
    }


def _output(artifact_id: str, file: Path, *, stage: str, workflow: str, task: str, kind: str, code_commit: str, expected_records: int | None = None) -> DeclaredOutput:
    return DeclaredOutput(
        artifact_id, str(file), kind=kind, stage=stage, expected_records=expected_records,
        provenance={"project_id": PROJECT_ID, "workflow_id": workflow, "task_id": task, "code_commit": code_commit},
        metadata={"stage": stage, "workflow": workflow},
    )


def _stage_run(*, stage: str, task: str, workflow: str, code_commit: str, artifact: DeclaredOutput, run_dir: Path, profile: Any, cwd: str, command: Sequence[str], base_config: Path | None = None, patches: Sequence[ConfigPatch] = ()) -> PreparedRun:
    arguments = [
        "{cwd}/scripts/robust_factorial_stage.py", "--stage", stage,
        "--artifact-id", artifact.artifact_id, "--artifact-path", artifact.path,
        "--kind", artifact.kind, "--project", PROJECT_ID, "--workflow", workflow,
        "--task", task, "--code-commit", code_commit,
    ]
    if artifact.expected_records is not None:
        arguments.extend(["--expected-records", str(artifact.expected_records)])
    arguments.extend(["--", *command])
    run = prepare_run(
        name=f"hk-atom14-{stage}-{task}", profile=profile, python_args=arguments,
        cwd=cwd, run_dir=str(run_dir), base_config=base_config, patches=patches,
        declared_outputs=[artifact],
    )
    if base_config is not None:
        _assert_rendered_config(run, base_config, patches)
    return run


def _cell_tasks(cell: Cell, *, workflow: str, code_commit: str, output_root: Path, cwd: str, profiles: Mapping[str, Any], parent_cells: Mapping[str, Mapping[str, Any]]) -> tuple[list[PreparedTask], dict[str, Any], tuple[str, Path, str]]:
    parent_config = _parent_config(cell, parent_cells)
    managed = output_root / "managed"
    tasks: list[PreparedTask] = []

    preflight_id = f"preflight-{cell.cell_id}"
    preflight_file = output_root / "preflight" / cell.cell_id / "report.json"
    preflight_output = _output(f"preflight/{cell.cell_id}/report.json", preflight_file, stage="preflight", workflow=workflow, task=preflight_id, kind="file", code_commit=code_commit, expected_records=1)
    preflight_run_dir = managed / "preflight" / cell.cell_id
    preflight_command = [
        "{cwd}/scripts/atom14_objective_preflight.py", "--config", "{config}",
        "--cell-id", cell.cell_id, "--output", str(preflight_file),
        "--seed", str(SIGMA_PROBE_SEED), "--sample-count", str(SIGMA_PROBE_SAMPLES),
        "--run-training-gate", "--expected-workers", "8",
        "--warmup-steps", "16", "--minimum-timed-rows", "16",
    ]
    if cell.objective:
        preflight_command.append("--objective-repair")
    preflight_run = _stage_run(
        stage="preflight", task=preflight_id, workflow=workflow, code_commit=code_commit,
        artifact=preflight_output, run_dir=preflight_run_dir, profile=profiles["gpu"], cwd=cwd,
        command=preflight_command, base_config=parent_config,
        patches=_preflight_patches(cell, preflight_run_dir),
    )
    tasks.append(PreparedTask(preflight_id, preflight_run, PREFLIGHT_RESOURCES, recovery=RECOVERY))

    train_id = f"train-{cell.cell_id}"
    train_dir = output_root / "train" / "L128" / cell.cell_id
    train_patches = _trainer_patches(cell, train_dir)
    train_run = prepare_run(
        name=f"hk-atom14-train-{cell.cell_id}-{code_commit[:7]}", profile=profiles["gpu"],
        python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
        cwd=cwd, run_dir=str(train_dir), base_config=parent_config, patches=train_patches,
    )
    _assert_rendered_config(train_run, parent_config, train_patches)
    diff = _resolved_diff(
        parent_config, _config_container(train_run), cell,
        child_sha256=hashlib.sha256(_config_artifact(train_run).content).hexdigest(),
        expected_parent_sha256=parent_cells[cell.architecture].get("sha256"),
    )
    tasks.append(PreparedTask(train_id, train_run, TRAIN_RESOURCES, wait_for=({"kind": "artifact", "task_id": preflight_id, "artifact_id": preflight_output.artifact_id},), recovery=RECOVERY))

    tag = f"step{CHECKPOINT_STEP:09d}"
    checkpoint = train_dir / f"{tag}.pt"
    sample_id = f"sample-{cell.cell_id}-{tag}"
    sample_dir = output_root / "samples" / tag / cell.cell_id
    sample_output = _output(f"samples/{tag}/{cell.cell_id}", sample_dir, stage="sample", workflow=workflow, task=sample_id, kind="directory", code_commit=code_commit, expected_records=SAMPLES_PER_LENGTH)
    sample_run_dir = managed / "sample" / cell.cell_id
    sample_run = _stage_run(
        stage="sample", task=sample_id, workflow=workflow, code_commit=code_commit,
        artifact=sample_output, run_dir=sample_run_dir, profile=profiles["gpu"], cwd=cwd,
        base_config=parent_config, patches=_evaluation_patches(cell, sample_run_dir),
        command=[
            "{cwd}/scripts/sample_short128_milestone.py", "--config", "{config}",
            "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
            "--lengths", "128", "--samples-per-length", str(SAMPLES_PER_LENGTH),
            "--batch-size", str(SAMPLE_BATCH_SIZE), "--seed", str(SAMPLE_SEED),
            "--precision", "bf16", "--compile",
        ],
    )
    tasks.append(PreparedTask(sample_id, sample_run, RESOURCES["sample"], wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": f"checkpoint/{tag}.pt"},), recovery=RECOVERY))

    esmfold_id = f"esmfold-{cell.cell_id}-{tag}"
    esmfold_dir = output_root / "esmfold" / tag / cell.cell_id / "L0128"
    esmfold_output = _output(f"esmfold/{tag}/{cell.cell_id}/L0128", esmfold_dir, stage="esmfold", workflow=workflow, task=esmfold_id, kind="directory", code_commit=code_commit, expected_records=SAMPLES_PER_LENGTH)
    esmfold_run = _stage_run(
        stage="esmfold", task=esmfold_id, workflow=workflow, code_commit=code_commit,
        artifact=esmfold_output, run_dir=managed / "esmfold" / cell.cell_id,
        profile=profiles["esmfold"], cwd=cwd,
        command=[
            "{cwd}/scripts/run_esmfold_designability_shard.py", "--sample-dir", str(sample_dir / "L0128"),
            "--output-dir", str(esmfold_dir), "--variant", cell.cell_id,
            "--step", str(CHECKPOINT_STEP), "--length", "128",
            "--expected-count", str(SAMPLES_PER_LENGTH), "--wrapper", "{cwd}/scripts/esmfold_predict_container",
            "--chunk-size", "8", "--bf16",
        ],
    )
    tasks.append(PreparedTask(esmfold_id, esmfold_run, RESOURCES["esmfold"], wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},), recovery=RECOVERY))

    analysis_id = f"analysis-{cell.cell_id}-{tag}"
    analysis_file = output_root / "analysis" / tag / cell.cell_id / "progres.json"
    analysis_output = _output(f"analysis/{tag}/{cell.cell_id}/progres.json", analysis_file, stage="analysis", workflow=workflow, task=analysis_id, kind="file", code_commit=code_commit, expected_records=1)
    analysis_run = _stage_run(
        stage="analysis", task=analysis_id, workflow=workflow, code_commit=code_commit,
        artifact=analysis_output, run_dir=managed / "analysis" / cell.cell_id,
        profile=profiles["progres"], cwd=cwd,
        command=[
            "{cwd}/scripts/analyze_progres_diversity.py", "--sample-dir", str(sample_dir / "L0128"),
            "--esmfold-dir", str(esmfold_dir), "--data-dir", str(PROGRES_DATA),
            "--expected-count", str(SAMPLES_PER_LENGTH), "--panel", cell.cell_id,
            "--step", str(CHECKPOINT_STEP), "--output", str(analysis_file),
        ],
    )
    tasks.append(PreparedTask(
        analysis_id, analysis_run, RESOURCES["cpu"],
        wait_for=(
            {"kind": "artifact", "task_id": esmfold_id, "artifact_id": esmfold_output.artifact_id},
            {"kind": "artifact", "task_id": "attest-progres-data", "artifact_id": "progres/attestation.json"},
        ),
        recovery=RECOVERY,
    ))
    return tasks, diff, (cell.cell_id, analysis_file, analysis_id)


def build_preflight_recovery(
    cell: Cell,
    *,
    code_commit: str,
    attempt: int,
    parent_cells: Mapping[str, Mapping[str, Any]] | None = None,
    output_root: Path | None = None,
) -> PreparedTask:
    """Rebuild one terminal preflight under its original workflow/task identity."""

    if attempt < 2:
        raise ValueError("preflight recovery attempt must be at least 2")
    parent_cells = PARENT_CELLS if parent_cells is None else parent_cells
    output_root = (
        REMOTE_RUN_ROOT / "atom14-causal-intervention-50k-L128" / code_commit
        if output_root is None
        else output_root
    )
    workflow = f"hk-atom14-causal-intervention-50k-L128-{code_commit[:7]}"
    task_id = f"preflight-{cell.cell_id}"
    report_file = output_root / "preflight" / cell.cell_id / "report.json"
    artifact = _output(
        f"preflight/{cell.cell_id}/report.json",
        report_file,
        stage="preflight",
        workflow=workflow,
        task=task_id,
        kind="file",
        code_commit=code_commit,
        expected_records=1,
    )
    run_dir = output_root / "managed" / "recovery" / task_id / f"attempt-{attempt}"
    command = [
        "{cwd}/scripts/atom14_objective_preflight.py",
        "--config", "{config}",
        "--cell-id", cell.cell_id,
        "--output", str(report_file),
        "--seed", str(SIGMA_PROBE_SEED),
        "--sample-count", str(SIGMA_PROBE_SAMPLES),
        "--run-training-gate",
        "--expected-workers", "8",
        "--warmup-steps", "16",
        "--minimum-timed-rows", "16",
    ]
    if cell.objective:
        command.append("--objective-repair")
    profile = _load_profile(GPU_PROFILE)
    run = _stage_run(
        stage="preflight",
        task=task_id,
        workflow=workflow,
        code_commit=code_commit,
        artifact=artifact,
        run_dir=run_dir,
        profile=profile,
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"),
        command=command,
        base_config=_parent_config(cell, parent_cells),
        patches=_preflight_patches(cell, run_dir),
    )
    return PreparedTask(task_id, run, PREFLIGHT_RESOURCES, recovery=RECOVERY)


def build_trainer_recovery(
    cell: Cell,
    *,
    code_commit: str,
    parent_cells: Mapping[str, Mapping[str, Any]] | None = None,
    output_root: Path | None = None,
) -> PreparedTask:
    """Rebuild one trainer for checkpoint-local resume under its original task."""

    parent_cells = PARENT_CELLS if parent_cells is None else parent_cells
    output_root = (
        REMOTE_RUN_ROOT / "atom14-causal-intervention-50k-L128" / code_commit
        if output_root is None
        else output_root
    )
    workflow = f"hk-atom14-causal-intervention-50k-L128-{code_commit[:7]}"
    preflight_id = f"preflight-{cell.cell_id}"
    task_id = f"train-{cell.cell_id}"
    train_dir = output_root / "train" / "L128" / cell.cell_id
    patches = _trainer_patches(cell, train_dir)
    run = prepare_run(
        name=f"hk-atom14-train-{cell.cell_id}-{code_commit[:7]}",
        profile=_load_profile(GPU_PROFILE),
        python_args=[
            "-m", "hierarchical_kaveh.train", "--config", "{config}",
            "--resume", "auto",
        ],
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"),
        run_dir=str(train_dir),
        base_config=_parent_config(cell, parent_cells),
        patches=patches,
    )
    _assert_rendered_config(run, _parent_config(cell, parent_cells), patches)
    return PreparedTask(
        task_id,
        run,
        TRAIN_RESOURCES,
        wait_for=({
            "kind": "artifact",
            "task_id": preflight_id,
            "artifact_id": f"preflight/{cell.cell_id}/report.json",
        },),
        recovery=RECOVERY,
    )


def build_workflow(code_commit: str, *, parent_cells: Mapping[str, Mapping[str, Any]] | None = None, output_root: Path | None = None) -> tuple[PreparedWorkflow, list[dict[str, Any]]]:
    parent_cells = PARENT_CELLS if parent_cells is None else parent_cells
    short = code_commit[:7]
    workflow = f"hk-atom14-causal-intervention-50k-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "atom14-causal-intervention-50k-L128" / code_commit if output_root is None else output_root
    cwd = str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}")
    profiles = {name: _load_profile(file) for name, file in {
        "gpu": GPU_PROFILE, "cpu": CPU_PROFILE, "esmfold": ESMFOLD_PROFILE, "progres": PROGRES_PROFILE,
    }.items()}
    tasks: list[PreparedTask] = []
    diffs: list[dict[str, Any]] = []
    analyses: list[tuple[str, Path, str]] = []

    attest_id = "attest-progres-data"
    attest_file = output_root / "progres" / "attestation.json"
    attest_output = _output("progres/attestation.json", attest_file, stage="attestation", workflow=workflow, task=attest_id, kind="file", code_commit=code_commit, expected_records=1)
    attest_run = _stage_run(
        stage="attestation", task=attest_id, workflow=workflow, code_commit=code_commit,
        artifact=attest_output, run_dir=output_root / "managed" / "attest-progres-data",
        profile=profiles["progres"], cwd=cwd,
        command=["{cwd}/scripts/attest_progres_data.py", "--data-dir", str(PROGRES_DATA), "--output", str(attest_file)],
    )
    tasks.append(PreparedTask(attest_id, attest_run, RESOURCES["cpu"], recovery=RECOVERY))

    for cell in CELLS:
        cell_tasks, diff, analysis = _cell_tasks(
            cell, workflow=workflow, code_commit=code_commit, output_root=output_root,
            cwd=cwd, profiles=profiles, parent_cells=parent_cells,
        )
        tasks.extend(cell_tasks)
        diffs.append(diff)
        analyses.append(analysis)

    aggregate_id = "aggregate"
    aggregate_file = output_root / "aggregate" / "step000050000" / "factorial.json"
    aggregate_output = _output("aggregate/step000050000/factorial.json", aggregate_file, stage="analysis", workflow=workflow, task=aggregate_id, kind="file", code_commit=code_commit, expected_records=1)
    aggregate_command = [
        "{cwd}/scripts/aggregate_atom14_causal_intervention.py", "--output", str(aggregate_file),
        "--workflow", workflow, "--step", str(CHECKPOINT_STEP), "--expected-cells", str(len(CELLS)),
        *[argument for cell_id, analysis_file, _ in analyses for argument in ("--analysis", f"{cell_id}={analysis_file}")],
    ]
    aggregate_run = _stage_run(
        stage="analysis", task=aggregate_id, workflow=workflow, code_commit=code_commit,
        artifact=aggregate_output, run_dir=output_root / "managed" / "aggregate",
        profile=profiles["cpu"], cwd=cwd, command=aggregate_command,
    )
    tasks.append(PreparedTask(
        aggregate_id, aggregate_run, RESOURCES["cpu"],
        wait_for=tuple(
            {"kind": "artifact", "task_id": task_id, "artifact_id": f"analysis/step{CHECKPOINT_STEP:09d}/{cell_id}/progres.json"}
            for cell_id, _file, task_id in analyses
        ),
        recovery=RECOVERY,
    ))
    workflow_object = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )
    if len(workflow_object.tasks) != 82:
        raise AssertionError(f"expected 82 tasks, built {len(workflow_object.tasks)}")
    return workflow_object, diffs


def _describe(workflow: PreparedWorkflow, diffs: list[dict[str, Any]], code_commit: str) -> dict[str, Any]:
    return {
        "schema": "hierarchical-kaveh.atom14-causal-intervention-launch.v1",
        "workflow_id": workflow.workflow_id, "request_id": workflow.request_id,
        "project_id": workflow.project_id, "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT, "scruffy_commit": SCRUFFY_COMMIT,
        "parent_workflow": PARENT_WORKFLOW, "parent_commit": PARENT_COMMIT,
        "parent_configs": {architecture: {"config": str(details["config"]), "sha256": details["sha256"]} for architecture, details in PARENT_CELLS.items()},
        "cells": [{"cell_id": cell.cell_id, "architecture": cell.architecture, "objective": cell.objective, "residual": cell.residual, "transport": cell.transport} for cell in CELLS],
        "invariants": {
            "data": {"metadata": str(METADATA), "min_length": 32, "max_length": 128, "mean_plddt_min": 80.0, "loop_length_max": 15, "loop_content_max": 0.4, "packing_density_min": 0.3, "batch_size_per_gpu": 256, "length_buckets": LENGTH_BUCKETS, "patch_capacities": PATCH_CAPACITIES, "shard_cache_size": None},
            "training": {"max_steps": MAX_STEPS, "checkpoint_every": TRAIN_CHECKPOINT_INTERVAL, "evaluate_at": CHECKPOINT_STEP, "self_conditioning_probability": 0.5, "ddp": False, "grad_accum": 1, "amp": "bf16", "compile_required": True, "fused_required": True, "optimizer_lr": 0.001},
        },
        "objective_repair": {**OBJECTIVE_VALUES, "c_out_weight_formula": "1 / c_out", "preflight_seed": SIGMA_PROBE_SEED, "preflight_sample_count": SIGMA_PROBE_SAMPLES},
        "evaluation": {"length": LENGTH, "samples": SAMPLES_PER_LENGTH, "sampler_batch_size": SAMPLE_BATCH_SIZE, "seed": SAMPLE_SEED, "progres_data": str(PROGRES_DATA), "same_fold_threshold": 0.8, "clustering": "complete linkage"},
        "task_count": len(workflow.tasks),
        "task_counts": dict(sorted(Counter(task.task_id.split("-", 1)[0] for task in workflow.tasks).items())),
        "config_diffs": diffs,
    }


REQUIRED_FEATURE_MARKERS = {
    "hierarchical_kaveh/config.py": ("smooth_lddt_sigma_max", "atom_ffn_residual_scale", "atom_to_residue_transport"),
    "hierarchical_kaveh/diffusion/losses.py": ("smooth_lddt_sigma_max", "smooth_lddt_resolved_atom_only", "smooth_lddt_c_out_compensation"),
    "hierarchical_kaveh/model/network.py": ("atom_ffn_residual_scale", "atom_to_residue_transport"),
    "hierarchical_kaveh/data/pipeline.py": ('"resolved_atom_mask": False',),
}


def _missing_paths(items: Sequence[Path]) -> list[str]:
    return [str(item) for item in items if not item.exists()]


def _validate_online(code_commit: str) -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    if _git("rev-parse", "HEAD", cwd=(REPO_ROOT / "external" / "koochak").resolve()) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected_checkout = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected_checkout:
        raise RuntimeError(f"run from the independent checkout {expected_checkout}")
    missing = _missing_paths((SCRUFFY_ROOT, SCRUFFY_SITE, METADATA, PROGRES_DATA))
    missing.extend(str(details["config"]) for details in PARENT_CELLS.values() if not Path(details["config"]).is_file())
    missing.extend(str(PROGRES_DATA / name) for name in ("trained_model.pt", "cath40.pt") if not (PROGRES_DATA / name).is_file())
    for relative, markers in REQUIRED_FEATURE_MARKERS.items():
        file = REPO_ROOT / relative
        if not file.is_file():
            missing.append(str(file))
            continue
        content = file.read_text(encoding="utf-8")
        missing.extend(f"{file}: marker {marker!r}" for marker in markers if marker not in content)
    if missing:
        raise RuntimeError(f"required launch inputs or feature implementation are missing: {missing}")
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    if not isinstance(allocation, Mapping) or allocation.get("state") != "running":
        raise RuntimeError("Scruffy allocation is not running")
    if allocation.get("controller_release") != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {allocation.get('controller_release')}")
    if snapshot.get("draining") is True or snapshot.get("launches_paused") is True:
        raise RuntimeError("Scruffy allocation is draining or launch-paused")
    return {"allocation_id": allocation.get("id"), "state": allocation.get("state"), "controller_release": allocation.get("controller_release"), "draining": snapshot.get("draining", False), "launches_paused": snapshot.get("launches_paused", False)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    allocation = None if args.dry_run else _validate_online(code_commit)
    workflow, diffs = build_workflow(code_commit)
    description = _describe(workflow, diffs, code_commit)
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return 0
    output_root = REMOTE_RUN_ROOT / "atom14-causal-intervention-50k-L128" / code_commit
    if output_root.exists():
        raise RuntimeError(f"refusing to reuse existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    diff_file = output_root / "resolved_config_diffs.json"
    diff_file.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "allocation": allocation, "diff_path": str(diff_file), "submission": submission}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
