#!/usr/bin/env python3
"""Prepare and, only after explicit gates, submit the four-cell Progres factorial."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import ConfigPatch, PreparedTask, PreparedWorkflow, load_environment_profile, prepare_run, submit_scruffy_workflow  # noqa: E402
from hierarchical_kaveh.data.progres import ProgresSidecarReader, sha256_file  # noqa: E402
from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    KOOCHAK_COMMIT, METADATA, MILESTONES, RECOVERY, REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT, RESOURCES, TRAIN_RESOURCES, _assert_config,
    _disabled_wandb, _git, _output, _stage_run, _tag,
)


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
SCRUFFY_COMMIT = "d9d89c45a232602aca2b7af790fde31a755b90a1"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-d60afabf-py310-cpython310-linux-x86_64/site")
SIDECAR_INDEX = Path("/mnt/lustre/users/kiarash-eitgbi/atom14/afdb_all_parsed/parsed_np_shards_with_ss_3di/progres_sidecars/progres-v1.1.0-128d-49830e1/index.json")
PROGRES_DATA = Path("/mnt/lustre/users/kiarash-eitgbi/code/progres-data/v1.1.0")
PARENT_COMMIT = "3ccc69aebb673ff556bcc59d2cba0fc0e4ddc6ba"
PARENT_WORKFLOW = "hk-patch-coarse-mixture-unconditioned-L128-3ccc69a"
PARENT_RUN_ROOT = REMOTE_RUN_ROOT / "patch-coarse-mixture-unconditioned-L128" / PARENT_COMMIT
LENGTH = 128
ARCHITECTURES = ("flat_after_node_no_transition", "pool_before_attention_pair_transition")
MIXTURES = (("mix50_50", 0.50, 0.50), ("mix75_25", 0.75, 0.25))
PARENT_CELLS = {
    (architecture, mixture_id): PARENT_RUN_ROOT / "train" / "L128" / f"{architecture}-{mixture_id}-sc0p5" / "config.yaml"
    for architecture in ARCHITECTURES
    for mixture_id, _, _ in MIXTURES
}
# Mixture values are inherited verbatim from the matched unconditioned parent.
MIXTURE_PATHS: set[str] = set()
CONDITION_PATHS = {
    "model.progres_conditioning", "model.progres_embedding_dim",
    "data.progres_sidecar_index_path", "train.progres_condition_dropout",
}
OUTPUT_PATHS = {"train.out_dir", "logging.csv_path", "logging.jsonl_path"}
MIXTURE_SHARD_CACHE_SIZE = 8
REQUIRED_TRAINER_GPUS = len(ARCHITECTURES)
MIN_REMAINING_SECONDS = TRAIN_RESOURCES[LENGTH]["time_limit_seconds"] + 3_600
HEARTBEAT_MAX_AGE_SECONDS = 120


@dataclass(frozen=True)
class Cell:
    architecture: str
    mixture_id: str
    strict_probability: float
    broader_probability: float

    @property
    def cell_id(self) -> str:
        return f"{self.architecture}-{self.mixture_id}-sc0p5-progres"


CELLS = tuple(Cell(architecture, name, strict, broad) for architecture in ARCHITECTURES for name, strict, broad in MIXTURES)


def _load_profile(source: Path):
    profile = load_environment_profile(source)
    variables = {key: value.replace("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site", str(SCRUFFY_SITE)) for key, value in profile.variables.items()}
    return replace(profile, profile_id=f"{profile.profile_id}-py310", variables=variables)


def _patches(cell: Cell, run_dir: Path, workflow: str) -> list[ConfigPatch]:
    del cell, workflow
    return [
        ConfigPatch("train.progres_condition_dropout", 0.5),
        ConfigPatch("model.progres_conditioning", True),
        ConfigPatch("model.progres_embedding_dim", 128),
        ConfigPatch("data.progres_sidecar_index_path", str(SIDECAR_INDEX)),
        ConfigPatch("logging.csv_path", str(run_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "log.jsonl")),
    ]


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        return {child_key: child_value for key, child in value.items() for child_key, child_value in _flatten(child, f"{prefix}.{key}" if prefix else str(key)).items()}
    return {prefix: value}


_MISSING = object()


def resolved_diff(parent_config: Path, child: Mapping[str, object], cell: Cell) -> dict[str, object]:
    parent = OmegaConf.to_container(OmegaConf.load(parent_config), resolve=True)
    parent_flat, child_flat = _flatten(parent), _flatten(child)
    if parent_flat.get("data.shard_cache_size") != MIXTURE_SHARD_CACHE_SIZE or child_flat.get("data.shard_cache_size") != MIXTURE_SHARD_CACHE_SIZE:
        raise AssertionError(
            f"conditioned cell {cell.cell_id} must inherit data.shard_cache_size={MIXTURE_SHARD_CACHE_SIZE}"
        )
    differences = []
    for key in sorted(set(parent_flat) | set(child_flat)):
        left, right = parent_flat.get(key, _MISSING), child_flat.get(key, _MISSING)
        if left is not _MISSING and right is not _MISSING and left == right:
            continue
        classification = "conditioning" if key in CONDITION_PATHS else "run_identity_or_output" if key in OUTPUT_PATHS else "unexpected"
        differences.append({"path": key, "parent": None if left is _MISSING else left, "child": None if right is _MISSING else right, "parent_present": left is not _MISSING, "child_present": right is not _MISSING, "classification": classification})
    observed = {item["path"] for item in differences}
    allowed = CONDITION_PATHS | OUTPUT_PATHS
    if observed != allowed:
        raise AssertionError(f"unexpected resolved-config differences for {cell.cell_id}: {sorted(observed ^ allowed)}")
    return {"cell_id": cell.cell_id, "architecture": cell.architecture, "mixture_id": cell.mixture_id, "parent_config": str(parent_config), "parent_config_sha256": hashlib.sha256(parent_config.read_bytes()).hexdigest(), "differences": differences, "allowed_paths": sorted(allowed)}


def _config_container(prepared) -> dict[str, object]:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    return OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)


def _cell_train(cell: Cell, workflow: str, output_root: Path, profile):
    parent_config = PARENT_CELLS[(cell.architecture, cell.mixture_id)]
    train_dir = output_root / "train" / "L128" / cell.cell_id
    patches = _patches(cell, train_dir, workflow)
    run = prepare_run(name=f"hk-mixture-conditioned-train-{cell.cell_id}-{_git('rev-parse', 'HEAD')[:7]}", profile=profile, python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"], cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"), run_dir=str(train_dir), base_config=parent_config, patches=patches)
    _assert_config(run, patches, base_config=parent_config)
    return run, train_dir, patches, parent_config


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, list[dict[str, object]]]:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-mixture-progres-conditioned-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-mixture-progres-conditioned-L128" / code_commit
    profiles = {"gpu": _load_profile(REPO_ROOT / "environments/tokyo-factorial-gpu.yaml"), "cpu": _load_profile(REPO_ROOT / "environments/tokyo-mixture-factorial-cpu.yaml"), "esmfold": _load_profile(REPO_ROOT / "environments/tokyo-factorial-esmfold.yaml")}
    bank_path = output_root / "condition_bank.json"
    bank_task = "condition-bank"
    bank_output = _output("condition-bank", bank_path, stage="analysis", workflow=workflow, task=bank_task, kind="file", expected_records=1)
    bank_run = _stage_run(stage="analysis", task=bank_task, workflow=workflow, artifact=bank_output, run_dir=output_root / "condition-bank.managed", profile=profiles["cpu"], command=["{cwd}/scripts/prepare_progres_condition_bank.py", "--index", str(SIDECAR_INDEX), "--metadata", str(METADATA), "--output", str(bank_path)])
    tasks = [PreparedTask(bank_task, bank_run, RESOURCES["analysis"], recovery=RECOVERY)]
    diffs: list[dict[str, object]] = []
    analysis_by_step: dict[int, list[tuple[str, str]]] = {step: [] for step in MILESTONES}
    for cell in CELLS:
        train, train_dir, patches, parent_config = _cell_train(cell, workflow, output_root, profiles["gpu"])
        diffs.append(resolved_diff(parent_config, _config_container(train), cell))
        train_id = f"train-{cell.cell_id}"
        tasks.append(PreparedTask(train_id, train, TRAIN_RESOURCES[LENGTH], wait_for=({"kind": "artifact", "task_id": bank_task, "artifact_id": bank_output.artifact_id},), recovery=RECOVERY))
        for step in MILESTONES:
            tag = _tag(step)
            checkpoint = train_dir / f"{tag}.pt"
            sample_id = f"sample-{cell.cell_id}-{tag}"
            sample_dir = output_root / "samples" / tag / cell.cell_id
            sample_output = _output(f"samples/{tag}/{cell.cell_id}", sample_dir, stage="sample", workflow=workflow, task=sample_id, kind="directory", expected_records=64)
            sample = _stage_run(stage="sample", task=sample_id, workflow=workflow, artifact=sample_output, run_dir=sample_dir.with_name(sample_dir.name + ".managed"), profile=profiles["gpu"], base_config=parent_config, patches=[*patches, *_disabled_wandb()], command=["{cwd}/scripts/sample_progres_conditioned_milestone.py", "--config", "{config}", "--checkpoint", str(checkpoint), "--condition-bank", str(bank_path), "--output-dir", str(sample_dir), "--seed", "20260905", "--batch-size", "4", "--precision", "bf16", "--compile"])
            tasks.append(PreparedTask(sample_id, sample, RESOURCES["sample"], wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": f"checkpoint/{tag}.pt"}, {"kind": "artifact", "task_id": bank_task, "artifact_id": bank_output.artifact_id}), recovery=RECOVERY))
            esm_id = f"esmfold-{cell.cell_id}-{tag}"
            esm_dir = output_root / "esmfold" / tag / cell.cell_id
            esm_output = _output(f"esmfold/{tag}/{cell.cell_id}", esm_dir, stage="esmfold", workflow=workflow, task=esm_id, kind="directory", expected_records=64)
            esm = _stage_run(stage="esmfold", task=esm_id, workflow=workflow, artifact=esm_output, run_dir=esm_dir.with_name(esm_dir.name + ".managed"), profile=profiles["esmfold"], command=["{cwd}/scripts/run_progres_conditioned_esmfold.py", "--sample-root", str(sample_dir), "--output-dir", str(esm_dir), "--wrapper", "{cwd}/scripts/esmfold_predict_container", "--chunk-size", "8", "--bf16"])
            tasks.append(PreparedTask(esm_id, esm, RESOURCES["esmfold"], wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},), recovery=RECOVERY))
            analysis_id = f"analysis-{cell.cell_id}-{tag}"
            analysis_path = output_root / "analysis" / tag / cell.cell_id / "analysis.json"
            analysis_output = _output(f"analysis/{tag}/{cell.cell_id}", analysis_path, stage="analysis", workflow=workflow, task=analysis_id, kind="file", expected_records=1)
            analysis = _stage_run(stage="analysis", task=analysis_id, workflow=workflow, artifact=analysis_output, run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"), profile=profiles["cpu"], command=["{cwd}/scripts/analyze_progres_conditioned_milestone.py", "--sample-dir", str(sample_dir), "--esmfold-dir", str(esm_dir), "--condition-bank", str(bank_path), "--sidecar-index", str(SIDECAR_INDEX), "--metadata", str(METADATA), "--data-dir", str(PROGRES_DATA), "--output", str(analysis_path)])
            tasks.append(PreparedTask(analysis_id, analysis, RESOURCES["analysis"], wait_for=({"kind": "artifact", "task_id": esm_id, "artifact_id": esm_output.artifact_id},), recovery=RECOVERY))
            analysis_by_step[step].append((cell.cell_id, str(analysis_path)))
    for step in MILESTONES:
        tag = _tag(step)
        task_id = f"aggregate-{tag}"
        output = output_root / "aggregate" / tag / "factorial.json"
        artifact = _output(f"aggregate/{tag}", output, stage="analysis", workflow=workflow, task=task_id, kind="file", expected_records=1)
        command = ["{cwd}/scripts/aggregate_progres_conditioned.py", "--step", str(step), "--output", str(output)]
        waits = []
        for cell_id, analysis_path in analysis_by_step[step]:
            command.extend(["--input", f"{cell_id}={analysis_path}"])
            waits.append({"kind": "artifact", "task_id": f"analysis-{cell_id}-{tag}", "artifact_id": f"analysis/{tag}/{cell_id}"})
        run = _stage_run(stage="analysis", task=task_id, workflow=workflow, artifact=artifact, run_dir=output.parent.with_name(output.parent.name + ".managed"), profile=profiles["cpu"], command=command)
        tasks.append(PreparedTask(task_id, run, RESOURCES["analysis"], wait_for=tuple(waits), recovery=RECOVERY))
    return PreparedWorkflow(request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow, project_id=PROJECT_ID, tasks=tuple(tasks)), diffs


def _status_value(*mappings: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for mapping in mappings:
        for key in keys:
            if key in mapping:
                return mapping[key]
    return None


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def validate_scruffy(snapshot: Mapping[str, object]) -> dict[str, object]:
    allocation = snapshot.get("allocation")
    if not isinstance(allocation, Mapping) or str(allocation.get("state", "")).lower() != "running":
        raise RuntimeError("Scruffy allocation is not RUNNING")
    draining = _status_value(allocation, snapshot, keys=("draining",))
    launches_paused = _status_value(allocation, snapshot, keys=("launches_paused", "launch_paused"))
    if draining is not False or launches_paused is not False:
        raise RuntimeError("Scruffy allocation is draining or launch-paused")
    age = _status_value(allocation, snapshot, keys=("heartbeat_age_seconds", "last_heartbeat_age_seconds"))
    if age is None:
        heartbeat = _parse_time(_status_value(allocation, snapshot, keys=("heartbeat_at", "last_heartbeat_at", "heartbeat")))
        if heartbeat is None:
            raise RuntimeError("Scruffy heartbeat is unavailable")
        age = max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())
    remaining = _status_value(allocation, snapshot, keys=("remaining_seconds", "remaining_time_seconds", "seconds_remaining"))
    if remaining is None:
        end = _parse_time(_status_value(allocation, snapshot, keys=("expires_at", "end_time", "deadline", "deadline_at")))
        remaining = None if end is None else (end - datetime.now(timezone.utc)).total_seconds()
    resources = allocation.get("resources", {})
    free_gpus = _status_value(resources if isinstance(resources, Mapping) else {}, allocation, snapshot, keys=("available_gpus", "free_gpus", "gpus_available"))
    if free_gpus is None:
        incarnation = allocation.get("incarnation")
        inventory = incarnation.get("inventory") if isinstance(incarnation, Mapping) else None
        total_gpus = sum(len(node.get("gpu_ids", ())) for node in inventory if isinstance(node, Mapping)) if isinstance(inventory, list) else 0
        active_states = {"running", "starting", "assigned", "launching", "submitted"}
        reserved_gpus = 0
        jobs = snapshot.get("jobs")
        if isinstance(jobs, Mapping):
            for job in jobs.values():
                if not isinstance(job, Mapping) or str(job.get("state", "")).lower() not in active_states:
                    continue
                assignment = job.get("assignment")
                reservations = assignment.get("reservations", ()) if isinstance(assignment, Mapping) else ()
                reserved_gpus += sum(len(item.get("gpu_ids", ())) for item in reservations if isinstance(item, Mapping))
        free_gpus = total_gpus - reserved_gpus
    if not isinstance(age, (int, float)) or float(age) > HEARTBEAT_MAX_AGE_SECONDS or not isinstance(remaining, (int, float)) or float(remaining) < MIN_REMAINING_SECONDS or not isinstance(free_gpus, (int, float)) or float(free_gpus) < REQUIRED_TRAINER_GPUS:
        raise RuntimeError("Scruffy allocation heartbeat, lifetime, or GPU capacity is insufficient")
    return {"allocation_id": str(allocation.get("allocation_id", allocation.get("id", ""))), "heartbeat_age_seconds": float(age), "remaining_seconds": float(remaining), "available_gpus": float(free_gpus), "controller_release": allocation.get("controller_release")}


def validate_online(code_commit: str) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    if _git("rev-parse", "HEAD", cwd=REPO_ROOT / "external/koochak") != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from independent checkout {expected}")
    for file in (METADATA, SIDECAR_INDEX):
        if not file.is_file():
            raise RuntimeError(f"required launch file is missing: {file}")
    if not (PROGRES_DATA / "trained_model.pt").is_file():
        raise RuntimeError(f"pinned Progres weights are missing: {PROGRES_DATA / 'trained_model.pt'}")
    if any(not config.is_file() for config in PARENT_CELLS.values()):
        raise RuntimeError("one or more immutable parent configs are missing")
    # The aggregate job already validated every sidecar.  Revalidate the
    # immutable index and all per-shard publications on the login node; the
    # first CPU condition-bank task performs the full row/embedding scan.
    reader = ProgresSidecarReader(SIDECAR_INDEX, metadata_path=METADATA, eager=False)
    if reader.index["progres"]["weights"]["sha256"] != "3fa3de9af77527da3efb8f2ee33ad05e678303d4e9cbe1f25a3916a106e56be3":
        raise RuntimeError("Progres weight checksum is not the pinned v1.1.0 checksum")
    if sha256_file(PROGRES_DATA / "trained_model.pt") != reader.index["progres"]["weights"]["sha256"]:
        raise RuntimeError("Progres weights do not match the sidecar aggregate identity")
    return {"sidecar_index": str(SIDECAR_INDEX), "sidecar_index_sha256": sha256_file(SIDECAR_INDEX), "sidecar": "fully validated", "weights_sha256": reader.index["progres"]["weights"]["sha256"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        sidecar = validate_online(code_commit)
    else:
        sidecar = {"sidecar": "not checked in dry-run"}
    workflow, diffs = build_workflow(code_commit)
    description = {"workflow_id": workflow.workflow_id, "request_id": workflow.request_id, "code_commit": code_commit, "task_count": len(workflow.tasks), "task_counts": {"condition_bank": 1, "trainers": 4, "sampling": 40, "esmfold": 40, "analysis": 40, "aggregate": 10}, "cells": [cell.cell_id for cell in CELLS], "sidecar": sidecar, "config_diffs": diffs}
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415
    snapshot = status(SCRUFFY_ROOT)
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("Scruffy status response is not a mapping")
    attestation = validate_scruffy(snapshot)
    if attestation["controller_release"] != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy release mismatch: expected {SCRUFFY_COMMIT}, got {attestation['controller_release']}")
    result = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "allocation": attestation, "submission": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
