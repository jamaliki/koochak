#!/usr/bin/env python3
"""Submit the unconditioned L128 strict/broader data-mixture factorial."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from hierarchical_kaveh.config import RunConfig  # noqa: E402

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    PreparedRun,
    PreparedTask,
    PreparedWorkflow,
    load_environment_profile,
    prepare_run,
    submit_scruffy_workflow,
)
from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    BASE_CONFIG,
    CPU_PROFILE,
    ESMFOLD_PROFILE,
    GPU_PROFILE,
    KOOCHAK_COMMIT,
    METADATA,
    MILESTONES,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    RESOURCES,
    SAMPLE_SEED,
    SAMPLES_PER_LENGTH,
    TRAIN_RESOURCES,
    TRAIN_CHECKPOINT_INTERVAL,
    VARIANTS,
    _assert_config,
    _disabled_wandb,
    _git,
    _output,
    _patches as parent_patches,
    _stage_run,
    _tag,
)


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
SCRUFFY_COMMIT = "8573c1c94986e017d6c8ad872930bdbd0abafc0f"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path(
    "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/"
    "scruffy-d60afabf-py310-cpython310-linux-x86_64/site"
)
PROGRES_DATA = Path("/mnt/lustre/users/kiarash-eitgbi/code/progres-data/v1.1.0")
PARENT_COMMIT = "97ce298cf0f5909ac0cbf50bdf94ab0481fbea8c"
PARENT_WORKFLOW = "hk-patch-coarse-factorial-500k-L128-followup-97ce298"
PARENT_RUN_ROOT = (
    REMOTE_RUN_ROOT
    / "patch-coarse-factorial-500k-followup"
    / PARENT_COMMIT
)
LENGTH = 128
ARCHITECTURES = (
    "flat_after_node_no_transition",
    "pool_before_attention_pair_transition",
)
# All four trainer cells may be admitted concurrently by Scruffy.
REQUIRED_TRAINER_GPUS = len(ARCHITECTURES) * 2
MIN_REMAINING_SECONDS = TRAIN_RESOURCES[LENGTH]["time_limit_seconds"] + 3_600
HEARTBEAT_MAX_AGE_SECONDS = 120
MIXTURES = (
    ("mix50_50", 0.50, 0.50),
    ("mix75_25", 0.75, 0.25),
)
MIXTURE_PATHS = {
    "data.mixture.strict_probability",
    "data.mixture.broader_probability",
    "data.mixture.broader_mean_plddt_min",
    "data.mixture.broader_loop_content_max",
    "data.mixture.broader_loop_length_max",
    "data.mixture.broader_packing_density_min",
    "data.mixture.broader_exclusive",
    "data.mixture.seed",
}
OUTPUT_DIFF_PATHS = {
    "train.out_dir",
    "logging.csv_path",
    "logging.jsonl_path",
}
OPERATIONAL_DIFF_PATHS = {"train.ckpt_every"}
MIXTURE_SHARD_CACHE_SIZE = None
PARENT_CELLS = {
    architecture: PARENT_RUN_ROOT / "train" / "L128" / f"{architecture}-strict-sc0p5" / "config.yaml"
    for architecture in ARCHITECTURES
}


@dataclass(frozen=True)
class Cell:
    architecture: str
    mixture_id: str
    strict_probability: float
    broader_probability: float

    @property
    def cell_id(self) -> str:
        return f"{self.architecture}-{self.mixture_id}-sc0p5"


CELLS = tuple(
    Cell(architecture, mixture_id, strict_probability, broader_probability)
    for architecture in ARCHITECTURES
    for mixture_id, strict_probability, broader_probability in MIXTURES
)


def _load_profile(source: Path):
    profile = load_environment_profile(source)
    variables = {
        key: value.replace(
            "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site",
            str(SCRUFFY_SITE),
        )
        for key, value in profile.variables.items()
    }
    return replace(
        profile,
        profile_id=f"{profile.profile_id}-py310",
        variables=variables,
    )


def _patches(cell: Cell, run_dir: Path, workflow: str) -> list[ConfigPatch]:
    variant = next(item for item in VARIANTS if item[0] == cell.architecture)
    patches = list(parent_patches(variant=variant, length=LENGTH, run_dir=run_dir, workflow=workflow))
    patches.extend(
        [
            ConfigPatch("train.self_conditioning_probability", 0.5),
            ConfigPatch("data.mixture.strict_probability", cell.strict_probability),
            ConfigPatch("data.mixture.broader_probability", cell.broader_probability),
            ConfigPatch("data.mixture.broader_mean_plddt_min", 80.0),
            ConfigPatch("data.mixture.broader_loop_content_max", 0.5),
            ConfigPatch("data.mixture.broader_loop_length_max", None),
            ConfigPatch("data.mixture.broader_packing_density_min", None),
            ConfigPatch("data.mixture.broader_exclusive", True),
            ConfigPatch("data.mixture.seed", 42),
            ConfigPatch("data.shard_cache_size", MIXTURE_SHARD_CACHE_SIZE),
        ]
    )
    return patches


def _config_container(prepared: PreparedRun) -> dict[str, object]:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    return OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        return {
            child_key: child_value
            for key, child in value.items()
            for child_key, child_value in _flatten(child, f"{prefix}.{key}" if prefix else str(key)).items()
        }
    return {prefix: value}


_MISSING = object()


def _resolved_config_with_defaults(config: object) -> dict[str, object]:
    defaults = OmegaConf.to_container(OmegaConf.structured(RunConfig), resolve=True)
    resolved = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(defaults), OmegaConf.create(config)),
        resolve=True,
    )
    if isinstance(resolved, dict) and isinstance(resolved.get("data"), dict):
        if resolved["data"].get("mixture") is None:
            # Optional sections absent from the parent are compared against
            # the child's concrete fields, not against a synthetic null node.
            resolved["data"].pop("mixture", None)
    if not isinstance(resolved, dict):
        raise TypeError("resolved configuration must be a mapping")
    return resolved


def _resolved_diff(parent_config: Path, child: dict[str, object], cell: Cell) -> dict[str, object]:
    if not parent_config.is_file():
        raise FileNotFoundError(parent_config)
    parent_flat = _flatten(_resolved_config_with_defaults(OmegaConf.load(parent_config)))
    child_flat = _flatten(_resolved_config_with_defaults(child))
    differences = []
    for key in sorted(set(parent_flat) | set(child_flat)):
        parent_value = parent_flat.get(key, _MISSING)
        child_value = child_flat.get(key, _MISSING)
        if (
            parent_value is not _MISSING
            and child_value is not _MISSING
            and parent_value == child_value
        ):
            continue
        differences.append(
            {
                "path": key,
                "parent": None if parent_value is _MISSING else parent_value,
                "child": None if child_value is _MISSING else child_value,
                "parent_present": parent_value is not _MISSING,
                "child_present": child_value is not _MISSING,
                "classification": (
                    "mixture"
                    if key in MIXTURE_PATHS
                    else "operational_cache"
                    if key in OPERATIONAL_DIFF_PATHS
                    else "run_identity_or_output"
                ),
            }
        )
    observed = {item["path"] for item in differences}
    expected = MIXTURE_PATHS | OPERATIONAL_DIFF_PATHS | OUTPUT_DIFF_PATHS
    if observed != expected:
        raise AssertionError(
            f"unexpected resolved-config differences for {cell.cell_id}: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )
    for key, expected_value in {
        "data.mixture.strict_probability": cell.strict_probability,
        "data.mixture.broader_probability": cell.broader_probability,
        "data.mixture.broader_mean_plddt_min": 80.0,
        "data.mixture.broader_loop_content_max": 0.5,
        "data.mixture.broader_loop_length_max": None,
        "data.mixture.broader_packing_density_min": None,
        "data.mixture.broader_exclusive": True,
        "data.mixture.seed": 42,
    }.items():
        if child_flat.get(key) != expected_value:
            raise AssertionError(f"unexpected mixture value {key} for {cell.cell_id}")
    return {
        "cell_id": cell.cell_id,
        "architecture": cell.architecture,
        "mixture_id": cell.mixture_id,
        "parent_config": str(parent_config),
        "parent_config_sha256": hashlib.sha256(parent_config.read_bytes()).hexdigest(),
        "differences": differences,
        "allowed_mixture_paths": sorted(MIXTURE_PATHS),
        "allowed_operational_paths": sorted(OPERATIONAL_DIFF_PATHS),
        "allowed_output_paths": sorted(OUTPUT_DIFF_PATHS),
    }


def _cell_train(cell: Cell, workflow: str, output_root: Path):
    train_dir = output_root / "train" / "L128" / cell.cell_id
    patches = _patches(cell, train_dir, workflow)
    run = prepare_run(
        name=f"hk-mixture-unconditioned-train-{cell.cell_id}-{_git('rev-parse', 'HEAD')[:7]}",
        profile=_load_profile(GPU_PROFILE),
        python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(train_dir), base_config=BASE_CONFIG, patches=patches,
    )
    _assert_config(run, patches)
    return run, train_dir, patches


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, list[dict[str, object]]]:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-mixture-unconditioned-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-mixture-unconditioned-L128" / code_commit
    profiles = {
        "gpu": _load_profile(GPU_PROFILE),
        "cpu": _load_profile(REPO_ROOT / "environments" / "tokyo-mixture-factorial-cpu.yaml"),
        "esmfold": _load_profile(ESMFOLD_PROFILE),
    }
    tasks: list[PreparedTask] = []
    diffs: list[dict[str, object]] = []
    for cell in CELLS:
        train, train_dir, patches = _cell_train(cell, workflow, output_root)
        diffs.append(_resolved_diff(PARENT_CELLS[cell.architecture], _config_container(train), cell))
        train_id = f"train-{cell.cell_id}"
        tasks.append(PreparedTask(train_id, train, TRAIN_RESOURCES[LENGTH], recovery=RECOVERY))
        for step in MILESTONES:
            tag = _tag(step)
            checkpoint = train_dir / f"{tag}.pt"
            checkpoint_id = f"checkpoint/{tag}.pt"
            sample_id = f"sample-{cell.cell_id}-{tag}"
            sample_dir = output_root / "samples" / tag / cell.cell_id
            sample_output = _output(
                f"samples/{tag}/{cell.cell_id}", sample_dir,
                stage="sample", workflow=workflow, task=sample_id,
                kind="directory", expected_records=SAMPLES_PER_LENGTH,
            )
            sample = _stage_run(
                stage="sample", task=sample_id, workflow=workflow, artifact=sample_output,
                run_dir=sample_dir.with_name(sample_dir.name + ".managed"), profile=profiles["gpu"],
                base_config=BASE_CONFIG, patches=[*patches, *_disabled_wandb()],
                command=[
                    "{cwd}/scripts/sample_short128_milestone.py", "--config", "{config}",
                    "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
                    "--lengths", "128", "--samples-per-length", str(SAMPLES_PER_LENGTH),
                    "--batch-size", "8", "--seed", str(SAMPLE_SEED), "--precision", "bf16", "--compile",
                ],
            )
            tasks.append(PreparedTask(
                sample_id, sample, RESOURCES["sample"],
                wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": checkpoint_id},),
                recovery=RECOVERY,
            ))
            esmfold_id = f"esmfold-{cell.cell_id}-{tag}"
            esmfold_dir = output_root / "esmfold" / tag / cell.cell_id / "L0128"
            esmfold_output = _output(
                f"esmfold/{tag}/{cell.cell_id}/L0128", esmfold_dir,
                stage="esmfold", workflow=workflow, task=esmfold_id,
                kind="directory", expected_records=SAMPLES_PER_LENGTH,
            )
            esmfold = _stage_run(
                stage="esmfold", task=esmfold_id, workflow=workflow, artifact=esmfold_output,
                run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"), profile=profiles["esmfold"],
                command=[
                    "{cwd}/scripts/run_esmfold_designability_shard.py",
                    "--sample-dir", str(sample_dir / "L0128"), "--output-dir", str(esmfold_dir),
                    "--variant", cell.cell_id, "--step", str(step), "--length", "128",
                    "--expected-count", str(SAMPLES_PER_LENGTH), "--wrapper", "{cwd}/scripts/esmfold_predict_container",
                    "--chunk-size", "8", "--bf16",
                ],
            )
            tasks.append(PreparedTask(
                esmfold_id, esmfold, RESOURCES["esmfold"],
                wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},),
                recovery=RECOVERY,
            ))
            progres_id = f"progres-{cell.cell_id}-{tag}"
            progres_path = output_root / "progres" / tag / cell.cell_id / "progres_diversity.json"
            progres_output = _output(
                f"progres/{tag}/{cell.cell_id}/progres_diversity.json", progres_path,
                stage="analysis", workflow=workflow, task=progres_id,
                kind="file", expected_records=1,
            )
            progres = _stage_run(
                stage="analysis", task=progres_id, workflow=workflow, artifact=progres_output,
                run_dir=progres_path.parent.with_name(progres_path.parent.name + ".managed"), profile=profiles["cpu"],
                command=[
                    "{cwd}/scripts/analyze_progres_diversity.py",
                    "--sample-dir", str(sample_dir / "L0128"), "--esmfold-dir", str(esmfold_dir),
                    "--data-dir", str(PROGRES_DATA), "--expected-count", str(SAMPLES_PER_LENGTH),
                    "--output", str(progres_path),
                ],
            )
            tasks.append(PreparedTask(
                progres_id, progres, RESOURCES["analysis"],
                wait_for=(
                    {"kind": "artifact", "task_id": esmfold_id, "artifact_id": esmfold_output.artifact_id},
                ), recovery=RECOVERY,
            ))
    variants = ",".join(cell.cell_id for cell in CELLS)
    for step in MILESTONES:
        tag = _tag(step)
        analysis_id = f"analysis-{tag}-L128"
        analysis_path = output_root / "analysis" / tag / "L128" / "milestone.json"
        analysis_output = _output(
            f"analysis/{tag}/L128/milestone.json", analysis_path,
            stage="analysis", workflow=workflow, task=analysis_id,
            kind="file", expected_records=1,
        )
        analysis = _stage_run(
            stage="analysis", task=analysis_id, workflow=workflow, artifact=analysis_output,
            run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"), profile=profiles["cpu"],
            command=[
                "{cwd}/scripts/analyze_esmfold_designability.py", "milestone",
                "--input-root", str(output_root / "esmfold" / tag), "--step", str(step),
                "--variants", variants, "--lengths", "128", "--expected-per-length", str(SAMPLES_PER_LENGTH),
                "--reference", CELLS[0].cell_id, "--output", str(analysis_path),
            ],
        )
        waits = tuple(
            {"kind": "artifact", "task_id": f"esmfold-{cell.cell_id}-{tag}", "artifact_id": f"esmfold/{tag}/{cell.cell_id}/L0128"}
            for cell in CELLS
        )
        tasks.append(PreparedTask(analysis_id, analysis, RESOURCES["analysis"], wait_for=waits, recovery=RECOVERY))
    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )
    return prepared, diffs


def _describe(workflow: PreparedWorkflow, diffs: list[dict[str, object]], code_commit: str) -> dict[str, object]:
    return {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "project_id": workflow.project_id,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "parents": {architecture: {"workflow_id": PARENT_WORKFLOW, "config": str(config)} for architecture, config in PARENT_CELLS.items()},
        "cells": [
            {"cell_id": cell.cell_id, "architecture": cell.architecture, "mixture_id": cell.mixture_id,
             "strict_probability": cell.strict_probability, "broader_probability": cell.broader_probability,
             "self_conditioning_probability": 0.5, "progres_conditioning": False}
            for cell in CELLS
        ],
        "data_contract": {
            "strict": {"min_length": 32, "max_length": 128, "mean_plddt_min": 80.0, "loop_length_max": 15, "loop_content_max": 0.4, "packing_density_min": 0.3},
            "broader_exclusive": {"min_length": 32, "max_length": 128, "mean_plddt_min": 80.0, "loop_length_max": None, "loop_content_max": 0.5, "packing_density_min": None, "excludes_strict": True},
        },
        "operational_cache": {
            "shard_cache_size_per_worker": MIXTURE_SHARD_CACHE_SIZE,
            "reason": (
                "preload each worker's disjoint physical-shard ownership set; "
                "the measured 124 GiB union fits the 240 GB cgroup"
            ),
        },
        "milestones": list(MILESTONES),
        "checkpoint_interval_steps": TRAIN_CHECKPOINT_INTERVAL,
        "task_count": len(workflow.tasks),
        "task_counts": {
            "trainers": len(CELLS), "sampling": len(CELLS) * len(MILESTONES),
            "esmfold": len(CELLS) * len(MILESTONES), "progres": len(CELLS) * len(MILESTONES),
            "designability_analysis": len(MILESTONES),
        },
        "mixture_resume_contract": (
            "deterministic_restart_from_canonical_source_stream; exact sample-level "
            "continuation is not claimed because length-bucket buffers and DataLoader "
            "prefetch state are not checkpointed"
        ),
        "config_diffs": diffs,
    }


def _status_value(*mappings: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for mapping in mappings:
        for key in keys:
            if key in mapping:
                return mapping[key]
    return None


def _status_flag(*mappings: Mapping[str, object], keys: tuple[str, ...]) -> bool | None:
    values = [mapping[key] for mapping in mappings for key in keys if key in mapping]
    if any(value is True for value in values):
        return True
    if any(value is False for value in values):
        return False
    return None


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_scruffy_snapshot(
    snapshot: Mapping[str, object],
    *,
    now: datetime | None = None,
    expected_allocation_id: str | None = None,
) -> dict[str, object]:
    """Attest the live allocation before allowing workflow submission."""

    allocation = snapshot.get("allocation")
    if not isinstance(allocation, Mapping):
        raise RuntimeError("Scruffy status has no allocation mapping")
    state = allocation.get("state")
    if not isinstance(state, str) or state.lower() != "running":
        raise RuntimeError(f"Scruffy allocation is not RUNNING: {state!r}")

    allocation_id = _status_value(
        allocation,
        snapshot,
        keys=("allocation_id", "allocation_identity", "id", "job_id"),
    )
    if allocation_id is None or not str(allocation_id).strip():
        raise RuntimeError("Scruffy status has no current allocation identity")
    allocation_id = str(allocation_id)
    if expected_allocation_id is not None and allocation_id != str(expected_allocation_id):
        raise RuntimeError(
            f"unexpected Scruffy allocation identity: expected {expected_allocation_id}, "
            f"got {allocation_id}"
        )

    for name, keys in {
        "draining": ("draining",),
        "launches_paused": ("launches_paused", "launch_paused"),
    }.items():
        value = _status_flag(allocation, snapshot, keys=keys)
        if value is not False:
            raise RuntimeError(f"Scruffy allocation {name} is not explicitly false: {value!r}")

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_value = _status_value(
        allocation,
        snapshot,
        keys=("heartbeat_age_seconds", "last_heartbeat_age_seconds"),
    )
    if age_value is not None:
        if not isinstance(age_value, (int, float)) or isinstance(age_value, bool) or not math.isfinite(float(age_value)):
            raise RuntimeError(f"invalid Scruffy heartbeat age: {age_value!r}")
        heartbeat_age = float(age_value)
    else:
        heartbeat_value = _status_value(
            allocation,
            snapshot,
            keys=("heartbeat_at", "last_heartbeat_at", "last_heartbeat", "heartbeat"),
        )
        heartbeat_at = _parse_timestamp(heartbeat_value)
        if heartbeat_at is None:
            raise RuntimeError("Scruffy status has no parseable heartbeat")
        heartbeat_age = max(0.0, (now - heartbeat_at).total_seconds())
    if heartbeat_age > HEARTBEAT_MAX_AGE_SECONDS:
        raise RuntimeError(f"Scruffy heartbeat is stale: {heartbeat_age:.1f}s")

    remaining_value = _status_value(
        allocation,
        snapshot,
        keys=("remaining_seconds", "remaining_time_seconds", "time_remaining_seconds", "seconds_remaining"),
    )
    if remaining_value is None:
        end_value = _status_value(
            allocation,
            snapshot,
            keys=("expires_at", "end_time", "deadline_at", "deadline", "allocation_end"),
        )
        end_at = _parse_timestamp(end_value)
        remaining_value = None if end_at is None else (end_at - now).total_seconds()
    if not isinstance(remaining_value, (int, float)) or isinstance(remaining_value, bool) or not math.isfinite(float(remaining_value)):
        raise RuntimeError("Scruffy status has no usable remaining lifetime")
    remaining_seconds = float(remaining_value)
    if remaining_seconds < MIN_REMAINING_SECONDS:
        raise RuntimeError(
            f"Scruffy allocation has insufficient remaining lifetime: {remaining_seconds:.0f}s "
            f"< {MIN_REMAINING_SECONDS}s"
        )

    resource_maps = [allocation, snapshot]
    for key in ("resources", "capacity", "available_resources"):
        value = allocation.get(key)
        if isinstance(value, Mapping):
            resource_maps.append(value)
        value = snapshot.get(key)
        if isinstance(value, Mapping):
            resource_maps.append(value)
    available_gpus = _status_value(
        *resource_maps,
        keys=("available_gpus", "gpus_available", "free_gpus", "gpus_free"),
    )
    if available_gpus is None:
        inventory = (allocation.get("incarnation") or {}).get("inventory")
        jobs = snapshot.get("jobs")
        if isinstance(inventory, list) and isinstance(jobs, Mapping):
            inventory_slots = {
                (item.get("name"), gpu_id)
                for item in inventory
                if isinstance(item, Mapping)
                for gpu_id in item.get("gpu_ids", ())
            }
            reserved_slots = {
                (reservation.get("node"), gpu_id)
                for job in jobs.values()
                if isinstance(job, Mapping) and job.get("state") == "running"
                for reservation in (
                    job.get("assignment") or job.get("last_assignment") or {}
                ).get("reservations", ())
                if isinstance(reservation, Mapping)
                for gpu_id in reservation.get("gpu_ids", ())
            }
            available_gpus = len(inventory_slots - reserved_slots)
    if not isinstance(available_gpus, (int, float)) or isinstance(available_gpus, bool) or not math.isfinite(float(available_gpus)):
        raise RuntimeError("Scruffy status has no usable available-GPU capacity")
    available_gpus = float(available_gpus)
    if available_gpus < REQUIRED_TRAINER_GPUS:
        raise RuntimeError(
            f"Scruffy allocation has insufficient free GPUs: {available_gpus:g} "
            f"< {REQUIRED_TRAINER_GPUS}"
        )
    return {
        "allocation_id": allocation_id,
        "state": "RUNNING",
        "draining": False,
        "launches_paused": False,
        "heartbeat_age_seconds": heartbeat_age,
        "remaining_seconds": remaining_seconds,
        "available_gpus": available_gpus,
        "controller_release": allocation.get("controller_release"),
    }


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    koochak_root = (REPO_ROOT / "external/koochak").resolve()
    if _git("rev-parse", "HEAD", cwd=koochak_root) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from independent checkout {expected}")
    missing = []
    if not SCRUFFY_ROOT.is_dir():
        missing.append(f"SCRUFFY_ROOT (directory): {SCRUFFY_ROOT}")
    if not SCRUFFY_SITE.is_dir():
        missing.append(f"SCRUFFY_SITE (directory): {SCRUFFY_SITE}")
    if not METADATA.is_file():
        missing.append(f"METADATA (file): {METADATA}")
    if not PROGRES_DATA.is_dir():
        missing.append(f"PROGRES_DATA (directory): {PROGRES_DATA}")
    missing.extend(str(config) for config in PARENT_CELLS.values() if not config.is_file())
    if missing:
        raise RuntimeError(f"required launch locations are missing: {missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _validate_online(code_commit)
    workflow, diffs = build_workflow(code_commit)
    description = _describe(workflow, diffs, code_commit)
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return
    output_root = REMOTE_RUN_ROOT / "patch-coarse-mixture-unconditioned-L128" / code_commit
    output_root.mkdir(parents=True, exist_ok=True)
    diff_path = output_root / "resolved_config_diffs.json"
    diff_path.write_text(json.dumps(description, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415
    snapshot = status(SCRUFFY_ROOT)
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("Scruffy status response is not a mapping")
    attestation = _validate_scruffy_snapshot(
        snapshot,
        expected_allocation_id=os.environ.get("SCRUFFY_ALLOCATION_ID"),
    )
    if attestation["controller_release"] != SCRUFFY_COMMIT:
        raise RuntimeError(
            f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, "
            f"got {attestation['controller_release']}"
        )
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "diff_path": str(diff_path), "allocation": attestation, "submission": submission}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
