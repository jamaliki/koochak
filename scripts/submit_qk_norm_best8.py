#!/usr/bin/env python3
"""Submit a matched per-head, non-affine QK-normalization panel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from omegaconf import OmegaConf

from scripts import submit_gain_invariant_factorial as shared


REPO_ROOT = shared.REPO_ROOT
REMOTE_RUN_ROOT = shared.REMOTE_RUN_ROOT
REMOTE_CODE_ROOT = shared.REMOTE_CODE_ROOT
SCRUFFY_ROOT = shared.SCRUFFY_ROOT
PROGRES_DATA = shared.PROGRES_DATA
GPU_PROFILE = shared.GPU_PROFILE
ESMFOLD_PROFILE = shared.ESMFOLD_PROFILE
PROGRES_PROFILE = shared.PROGRES_PROFILE
PROJECT_ID = shared.PROJECT_ID
KOOCHAK_COMMIT = shared.KOOCHAK_COMMIT
SCRUFFY_COMMIT = shared.SCRUFFY_COMMIT
TRAIN_RESOURCES = shared.TRAIN_RESOURCES
PREFLIGHT_RESOURCES = shared.PREFLIGHT_RESOURCES
SAMPLE_RESOURCES = shared.SAMPLE_RESOURCES
ESMFOLD_RESOURCES = shared.ESMFOLD_RESOURCES
ANALYSIS_RESOURCES = shared.ANALYSIS_RESOURCES
RECOVERY = shared.RECOVERY
ACK_TIMEOUT_SECONDS = 300

OBJECTIVE_ROOT = REMOTE_RUN_ROOT / "objective-decomp-continuations-500k-L128" / "0db43cf38f3a8d9569044a30e5731d81af661536" / "train/L128"
DIT_ROOT = REMOTE_RUN_ROOT / "dit-adaln-zero-counterparts-L128" / "142664f5526bb8f14da6f0570b25598be5e8d4cc" / "train/L128"
QK_KEY = "model.qk_norm_mode"
QK_VALUE = "per_head_rms"
OUTPUT_PATHS = {"train.out_dir", "logging.csv_path", "logging.jsonl_path"}
ALLOWED_DIFF_PATHS = {*OUTPUT_PATHS, QK_KEY}
SAMPLES = 32
SAMPLE_BATCH_SIZE = 8
SAMPLE_SEED = 20260901


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    family: str
    parent_config: Path
    parent_sha256: str
    max_steps: int = 500_000

    @property
    def milestones(self) -> tuple[int, ...]:
        return tuple(range(50_000, self.max_steps + 1, 50_000))


def _objective_cell(cell_id: str, sha256: str) -> Cell:
    return Cell(cell_id, "objective_decomposition_continuation_500k", OBJECTIVE_ROOT / cell_id / "config.yaml", sha256)


def _dit_cell(cell_id: str, sha256: str) -> Cell:
    return Cell(cell_id, "dit_depth_sandwich_500k", DIT_ROOT / cell_id / "config.yaml", sha256)


CELLS = (
    _objective_cell("lddt-m1g0c1", "1f4bc5203b2fa8cfa708e0d8c85e2531069a9ed49c51df678a5a0ef9610d0504"),
    _objective_cell("lddt-m0g1c1", "48f26191c3e8f5b0c07a7b9f9699106b07d863598eb4b8608972d5b4ef50d6dd"),
    _objective_cell("lddt-m1g1c1", "55f1d056104f9cf889f0b012949959fec4c1cbb94ea5ea6f7dc28ba12e735179"),
    _objective_cell("lddt-m0g1c0", "46319ec58c89248569276a034ec75dcbe9731e71c6da3a7c89e976976064f7d7"),
    _objective_cell("lddt-m1g0c0", "488d28766c8ebba861ae3866a29949f87c894023d1274b3d4c89db6ad72eab6a"),
    _objective_cell("lddt-m0g0c1", "781e0ec7ff4005382c9ed4a5ef68e83a4f1d86f628bbb44b05549fa071f6c7f4"),
    _objective_cell("lddt-m1g1c0", "33ca66521dfebc72d27cdba76cbda7a22eeaf0a20d8374a88be756fedae9823a"),
    _dit_cell("flat_after_node_no_transition-depth-attn-sandwich", "1c1ba110979f6306bd5f551666a53877e37b91951d3fab59ebefff030a93ac75"),
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _patches(run_dir: Path, *, preflight: bool = False) -> list[shared.ConfigPatch]:
    patches = [
        shared.ConfigPatch(QK_KEY, QK_VALUE),
        shared.ConfigPatch("logging.csv_path", str(run_dir / "metrics.csv")),
        shared.ConfigPatch("logging.jsonl_path", str(run_dir / "metrics.jsonl")),
    ]
    if preflight:
        patches.extend([
            shared.ConfigPatch("train.max_steps", 64),
            shared.ConfigPatch("train.log_every", 1),
            shared.ConfigPatch("train.save_final", False),
        ])
    return patches


def _at(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise AssertionError(f"required config path is absent: {dotted}")
        value = value[part]
    return value


def _resolved_diff(cell: Cell, prepared: shared.PreparedRun) -> dict[str, Any]:
    parent_bytes = cell.parent_config.read_bytes()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    if parent_hash != cell.parent_sha256:
        raise AssertionError(f"immutable parent hash mismatch for {cell.cell_id}: {parent_hash}")
    parent = OmegaConf.to_container(OmegaConf.load(cell.parent_config), resolve=True)
    child = shared.robust._config_container(prepared)
    if not isinstance(parent, dict):
        raise TypeError("parent config must be a mapping")
    parent_flat, child_flat = shared._flatten(parent), shared._flatten(child)
    differences = []
    for dotted in sorted(set(parent_flat) | set(child_flat)):
        parent_present, child_present = dotted in parent_flat, dotted in child_flat
        before, after = parent_flat.get(dotted), child_flat.get(dotted)
        if parent_present == child_present and before == after:
            continue
        differences.append({
            "path": dotted,
            "parent_present": parent_present,
            "child_present": child_present,
            "parent": before,
            "child": after,
            "classification": "scientific_factor" if dotted == QK_KEY else "run_identity_or_output",
        })
    observed = {item["path"] for item in differences}
    if observed != ALLOWED_DIFF_PATHS:
        raise AssertionError(
            f"unexpected resolved diff for {cell.cell_id}: observed={sorted(observed)}, "
            f"expected={sorted(ALLOWED_DIFF_PATHS)}"
        )
    if _at(child, QK_KEY) != QK_VALUE or _at(child, "train.max_steps") != cell.max_steps:
        raise AssertionError(f"{cell.cell_id}: QK patch or 500k horizon was not applied")
    artifact = shared.robust._config_artifact(prepared)
    return {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "parent_config": str(cell.parent_config),
        "parent_config_sha256": parent_hash,
        "child_config_sha256": hashlib.sha256(artifact.content).hexdigest(),
        "differences": differences,
        "unexpected_differences": [],
    }


def _build_workflow(code_commit: str) -> tuple[shared.PreparedWorkflow, dict[str, Any]]:
    short = code_commit[:7]
    workflow = f"hk-qk-norm-best8-L128-{short}-v2"
    output_root = REMOTE_RUN_ROOT / "qk-norm-best8-L128" / code_commit / "v2"
    cwd = str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}")
    profiles = {
        "gpu": shared._profile(GPU_PROFILE),
        "esmfold": shared._profile(ESMFOLD_PROFILE),
        "progres": shared._profile(PROGRES_PROFILE),
    }
    tasks: list[shared.PreparedTask] = []
    diffs: list[dict[str, Any]] = []

    # The shared evaluator is deliberately reused, but its patch function is swapped
    # only within this process so every train/sample config gets the same QK-only diff.
    old_patches = shared._patches
    shared._patches = _patches
    try:
        for cell in CELLS:
            preflight_id = f"preflight-{cell.cell_id}"
            preflight_file = output_root / "preflight" / cell.cell_id / "report.json"
            preflight_output = shared.robust._output(
                f"preflight/{cell.cell_id}/report.json", preflight_file, stage="preflight",
                workflow=workflow, task=preflight_id, kind="file", code_commit=code_commit, expected_records=1,
            )
            preflight_dir = output_root / "managed/preflight" / cell.cell_id
            preflight_run = shared.robust._stage_run(
                stage="preflight", task=preflight_id, workflow=workflow, code_commit=code_commit,
                artifact=preflight_output, run_dir=preflight_dir, profile=profiles["gpu"], cwd=cwd,
                base_config=cell.parent_config, patches=_patches(preflight_dir, preflight=True),
                command=[
                    "{cwd}/scripts/stability_preflight.py", "--config", "{config}", "--output", str(preflight_file),
                    "--cell-id", cell.cell_id, "--expected-workers", "8", "--warmup-steps", "16",
                    "--minimum-timed-rows", "16",
                ],
            )
            tasks.append(shared.PreparedTask(preflight_id, preflight_run, PREFLIGHT_RESOURCES, recovery=RECOVERY))

            train_id = f"train-{cell.cell_id}"
            train_dir = output_root / "train/L128" / cell.cell_id
            train_patches = _patches(train_dir)
            train_run = shared.prepare_run(
                name=f"hk-qk-norm-train-{cell.cell_id}-{short}", profile=profiles["gpu"],
                python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
                cwd=cwd, run_dir=str(train_dir), base_config=cell.parent_config, patches=train_patches,
            )
            shared.robust._assert_rendered_config(train_run, cell.parent_config, train_patches)
            diffs.append(_resolved_diff(cell, train_run))
            tasks.append(shared.PreparedTask(
                train_id, train_run, TRAIN_RESOURCES,
                wait_for=(
                    {"kind": "artifact", "task_id": preflight_id, "artifact_id": preflight_output.artifact_id},
                ), recovery=RECOVERY,
            ))
            tasks.extend(shared._evaluation_tasks(
                cell, workflow=workflow, code_commit=code_commit, output_root=output_root, cwd=cwd,
                profiles=profiles, train_id=train_id, train_dir=train_dir,
                attestation_artifact=None,
            ))
    finally:
        shared._patches = old_patches

    prepared = shared.PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )
    expected_tasks = len(CELLS) * (2 + 3 * len(CELLS[0].milestones))
    if len(prepared.tasks) != expected_tasks:
        raise AssertionError(f"expected {expected_tasks} tasks, built {len(prepared.tasks)}")
    description = {
        "schema": "hierarchical-kaveh.qk-norm-best8.v1",
        "workflow_id": workflow,
        "request_id": prepared.request_id,
        "project_id": PROJECT_ID,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "scientific_intervention": {
            "path": QK_KEY,
            "value": QK_VALUE,
            "formulation": "per-head parameter-free RMS normalization of Q and K",
            "fixed_logit_scale": "1/sqrt(head_dim)",
            "all_parent_architecture_and_gain_paths_preserved": True,
        },
        "progres_data_contract": {
            "data_dir": str(PROGRES_DATA),
            "attestation_task": "omitted to remain within Scruffy's 256-task workflow limit",
            "analysis_inputs_are_pinned_and_read_only": True,
        },
        "selection_basis": "top eight 250k ledger arms, including the 17/32 objective tie and DiT winner",
        "cells": [
            {
                "cell_id": cell.cell_id,
                "family": cell.family,
                "max_steps": cell.max_steps,
                "milestones": list(cell.milestones),
                "parent_config": str(cell.parent_config),
                "trainer_task_id": f"train-{cell.cell_id}",
                "from_scratch": True,
            }
            for cell in CELLS
        ],
        "task_count": len(prepared.tasks),
        "task_counts": {
            "attestation": 0,
            "preflights": len(CELLS),
            "trainers": len(CELLS),
            "sampling": len(CELLS) * len(CELLS[0].milestones),
            "esmfold": len(CELLS) * len(CELLS[0].milestones),
            "progres_analysis": len(CELLS) * len(CELLS[0].milestones),
        },
        "samples": {"length": 128, "count": SAMPLES, "batch_size": SAMPLE_BATCH_SIZE, "seed": SAMPLE_SEED},
        "cache_contract": {"shard_cache_size": None, "resident_disjoint_worker_ownership": True},
        "artifact_ack_timeout_seconds": ACK_TIMEOUT_SECONDS,
        "resolved_config_diffs": diffs,
    }
    return prepared, {"description": description, "output_root": output_root}


def _validate_online(code_commit: str) -> dict[str, Any]:
    # Reuse the hardened online checks, with this panel's immutable parent set.
    old_cells = shared.CELLS
    shared.CELLS = CELLS
    try:
        return shared._validate_online(code_commit)
    finally:
        shared.CELLS = old_cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    allocation = None if args.dry_run else _validate_online(code_commit)
    workflow, bundle = _build_workflow(code_commit)
    description, output_root = bundle["description"], bundle["output_root"]
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return 0
    if output_root.exists():
        raise RuntimeError(f"refusing to reuse existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    diff_file = output_root / "resolved_config_diffs.json"
    diff_file.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    submission = shared.submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({
        "workflow": description,
        "allocation": allocation,
        "diff_path": str(diff_file),
        "submission": submission,
    }, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
