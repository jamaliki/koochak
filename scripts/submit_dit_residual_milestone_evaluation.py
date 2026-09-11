#!/usr/bin/env python3
"""Evaluate the missing 200k-350k milestones for the clean-residual DiT panel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import PreparedTask, PreparedWorkflow, submit_scruffy_workflow  # noqa: E402
from scripts import submit_gain_invariant_factorial as shared  # noqa: E402


robust = shared.robust
PROJECT_ID = shared.PROJECT_ID
KOOCHAK_COMMIT = shared.KOOCHAK_COMMIT
SCRUFFY_COMMIT = shared.SCRUFFY_COMMIT
SCRUFFY_ROOT = shared.SCRUFFY_ROOT
SCRUFFY_SITE = robust.SCRUFFY_SITE
REMOTE_CODE_ROOT = shared.REMOTE_CODE_ROOT
REMOTE_RUN_ROOT = shared.REMOTE_RUN_ROOT
PROGRES_DATA = shared.PROGRES_DATA
RECOVERY = shared.RECOVERY
RESOURCES = {
    "sample": shared.SAMPLE_RESOURCES,
    "esmfold": shared.ESMFOLD_RESOURCES,
    "cpu": shared.ANALYSIS_RESOURCES,
}
SAMPLE_SEED = 20260910
MILESTONES = (200_000, 250_000, 300_000, 350_000)
OUTPUT_ROOT_NAME = "dit-residual-milestone-evaluation-50k"


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    source_commit: str
    arm: str
    config_sha256: str

    @property
    def parent_config(self) -> Path:
        return (
            REMOTE_RUN_ROOT / "dit-clean-residual-optimizer-500k" / self.source_commit
            / "v1/train/L128" / self.arm / "config.yaml"
        )

    @property
    def milestones(self) -> tuple[int, ...]:
        return MILESTONES


CELLS = (
    Cell("astraF1ac2b1-adam", "f1ac2b132147af1e448928d65d35bc0e7fc5fdc8", "adam", "a39ce3d42624d7aa2ba190ad480296ef5746a8a2d29209bdf463fae64c6f0c32"),
    Cell("astraF1ac2b1-adamw", "f1ac2b132147af1e448928d65d35bc0e7fc5fdc8", "adamw", "2fd97965e64b4b927c4c4f016ac4b5dfd7dfa84328f3e77015f309d13bc35e0a"),
    Cell("clean4458e63-adam", "4458e63bf39361d12e2392957d81ef064c0fc009", "adam", "5260fb388e8ca5a3678080748341d9342a7e31078d03ba1ef93d7e73116fe17a"),
    Cell("clean4458e63-adamw", "4458e63bf39361d12e2392957d81ef064c0fc009", "adamw", "d23215218ecb9259fa7c5e2fb247f9d521bb353ecb1bffe3d0c595ce6526d2dd"),
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _tag(step: int) -> str:
    return f"step{step:09d}"


def _evaluation_patches(run_dir: Path) -> list[Any]:
    return [
        shared.ConfigPatch("logging.csv_path", str(run_dir / "metrics.csv")),
        shared.ConfigPatch("logging.jsonl_path", str(run_dir / "metrics.jsonl")),
    ]


def _validate_inputs() -> None:
    missing: list[str] = []
    for item in (SCRUFFY_ROOT, SCRUFFY_SITE, PROGRES_DATA):
        if not item.exists():
            missing.append(str(item))
    for cell in CELLS:
        if not cell.parent_config.is_file():
            missing.append(str(cell.parent_config))
            continue
        observed = hashlib.sha256(cell.parent_config.read_bytes()).hexdigest()
        if observed != cell.config_sha256:
            raise RuntimeError(f"config hash mismatch for {cell.cell_id}: {observed}")
        for step in MILESTONES:
            checkpoint = cell.parent_config.parent / f"{_tag(step)}.pt"
            ready = checkpoint.with_name(checkpoint.name + ".ready.json")
            if not checkpoint.is_file():
                missing.append(str(checkpoint))
            elif not ready.is_file():
                missing.append(str(ready))
    for name in ("trained_model.pt", "cath40.pt"):
        if not (PROGRES_DATA / name).is_file():
            missing.append(str(PROGRES_DATA / name))
    if missing:
        raise RuntimeError("missing evaluation inputs: " + ", ".join(missing))


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, dict[str, Any]]:
    short = code_commit[:7]
    workflow = f"hk-dit-residual-milestone-evaluation-50k-{short}-v2"
    output_root = REMOTE_RUN_ROOT / OUTPUT_ROOT_NAME / code_commit / "v2"
    cwd = str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}")
    profiles = {
        "gpu": shared._profile(shared.GPU_PROFILE),
        "esmfold": shared._profile(shared.ESMFOLD_PROFILE),
        "progres": shared._profile(shared.PROGRES_PROFILE),
    }
    old_patches, old_seed = shared._patches, shared.SAMPLE_SEED
    shared._patches = _evaluation_patches
    shared.SAMPLE_SEED = SAMPLE_SEED
    try:
        tasks: list[PreparedTask] = []
        for cell in CELLS:
            tasks.extend(
                shared._evaluation_tasks(
                    cell,
                    workflow=workflow,
                    code_commit=code_commit,
                    output_root=output_root,
                    cwd=cwd,
                    profiles=profiles,
                    train_id=f"existing-train-{cell.cell_id}",
                    train_dir=cell.parent_config.parent,
                    attestation_artifact=None,
                )
            )
    finally:
        shared._patches, shared.SAMPLE_SEED = old_patches, old_seed
    if len(tasks) != 48:
        raise AssertionError(f"expected 48 tasks, built {len(tasks)}")
    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )
    description = {
        "schema": "hierarchical-kaveh.dit-residual-milestone-evaluation.v2",
        "workflow_id": workflow,
        "request_id": prepared.request_id,
        "project_id": PROJECT_ID,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "milestones_submitted": list(MILESTONES),
        "previously_evaluated_milestones": [50_000, 100_000, 150_000],
        "cells": [
            {
                "cell_id": cell.cell_id,
                "source_commit": cell.source_commit,
                "arm": cell.arm,
                "config": str(cell.parent_config),
                "config_sha256": cell.config_sha256,
            }
            for cell in CELLS
        ],
        "sampling": {
            "length": 128,
            "samples": 32,
            "batch_size": 8,
            "seed": SAMPLE_SEED,
            "weights": "ema",
            "precision": "bf16",
            "compile": True,
        },
        "task_counts": {"sampling": 16, "esmfold": 16, "progres_analysis": 16},
        "checkpoint_inputs_are_durable_at_submission": True,
    }
    return prepared, {"description": description, "output_root": output_root}


def _validate_online(code_commit: str) -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    expected_checkout = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected_checkout:
        raise RuntimeError(f"run from the independent checkout {expected_checkout}")
    if _git("rev-parse", "HEAD", cwd=REPO_ROOT / "external/koochak") != KOOCHAK_COMMIT:
        raise RuntimeError(f"Koochak commit mismatch: expected {KOOCHAK_COMMIT}")
    _validate_inputs()
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    if not isinstance(allocation, Mapping) or allocation.get("state") != "running":
        raise RuntimeError("Scruffy allocation is not running")
    if allocation.get("controller_release") != SCRUFFY_COMMIT:
        raise RuntimeError("Scruffy controller release mismatch")
    if snapshot.get("draining") is True or snapshot.get("launches_paused") is True:
        raise RuntimeError("Scruffy allocation is draining or launch-paused")
    return {"allocation_id": allocation.get("id"), "controller_release": allocation.get("controller_release")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    workflow, bundle = build_workflow(code_commit)
    description, output_root = bundle["description"], bundle["output_root"]
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return 0
    allocation = _validate_online(code_commit)
    if output_root.exists():
        raise RuntimeError(f"refusing to reuse existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = output_root / "launch_manifest.json"
    manifest.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"allocation": allocation, "manifest": str(manifest), "submission": submission}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
