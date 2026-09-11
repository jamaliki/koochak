#!/usr/bin/env python3
"""Evaluate missing 50k-spaced checkpoints across the current architecture wave.

This launcher intentionally covers more than the clean-residual optimizer panel:
the new Muon campaigns and the signal/residual-shell factorial are included too.
Existing evaluation trees are not reused; every task writes to a fresh, immutable
workflow root and validates its durable checkpoint inputs before submission.
"""

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
SCRUFFY_COMMIT = "00c2f09468372fe138fd7d7878e6dfa67fb8c29b"
SCRUFFY_ROOT = shared.SCRUFFY_ROOT
SCRUFFY_SITE = Path(
    "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/"
    f"scruffy-{SCRUFFY_COMMIT}-py310-cpython310/site"
)
REMOTE_CODE_ROOT = shared.REMOTE_CODE_ROOT
REMOTE_RUN_ROOT = shared.REMOTE_RUN_ROOT
PROGRES_DATA = shared.PROGRES_DATA
RECOVERY = shared.RECOVERY
SAMPLE_SEED = 20260910
OUTPUT_ROOT_NAME = "architecture-milestone-evaluation-50k-20260911"
RETRY_OUTPUT_ROOT_NAME = "architecture-milestone-evaluation-50k-retry-20260911"
PREVIOUSLY_EVALUATED = {
    "clean-residual": (50_000, 100_000, 150_000),
}


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    family: str
    source_commit: str
    parent_config: Path
    config_sha256: str
    milestones: tuple[int, ...]


def _config(campaign: str, source_commit: str, arm: str) -> Path:
    return REMOTE_RUN_ROOT / campaign / source_commit / "v1/train" / arm / "config.yaml"


CELLS = (
    Cell(
        "astraF1ac2b1-adam", "clean-residual", "f1ac2b132147af1e448928d65d35bc0e7fc5fdc8",
        _config("dit-clean-residual-optimizer-500k", "f1ac2b132147af1e448928d65d35bc0e7fc5fdc8", "L128/adam"),
        "a39ce3d42624d7aa2ba190ad480296ef5746a8a2d29209bdf463fae64c6f0c32",
        (200_000, 250_000, 300_000, 350_000),
    ),
    Cell(
        "astraF1ac2b1-adamw", "clean-residual", "f1ac2b132147af1e448928d65d35bc0e7fc5fdc8",
        _config("dit-clean-residual-optimizer-500k", "f1ac2b132147af1e448928d65d35bc0e7fc5fdc8", "L128/adamw"),
        "2fd97965e64b4b927c4c4f016ac4b5dfd7dfa84328f3e77015f309d13bc35e0a",
        (200_000, 250_000, 300_000, 350_000),
    ),
    Cell(
        "clean4458e63-adam", "clean-residual", "4458e63bf39361d12e2392957d81ef064c0fc009",
        _config("dit-clean-residual-optimizer-500k", "4458e63bf39361d12e2392957d81ef064c0fc009", "L128/adam"),
        "5260fb388e8ca5a3678080748341d9342a7e31078d03ba1ef93d7e73116fe17a",
        (200_000, 250_000, 300_000, 350_000),
    ),
    Cell(
        "clean4458e63-adamw", "clean-residual", "4458e63bf39361d12e2392957d81ef064c0fc009",
        _config("dit-clean-residual-optimizer-500k", "4458e63bf39361d12e2392957d81ef064c0fc009", "L128/adamw"),
        "d23215218ecb9259fa7c5e2fb247f9d521bb353ecb1bffe3d0c595ce6526d2dd",
        (200_000, 250_000, 300_000, 350_000),
    ),
    Cell(
        "muon-cbeta-slot4", "muon-cbeta-clock", "882c5a1b569c1f3eb8bc90595ab0af124df3919d",
        _config("l128-batch256-muon-cbeta-clock-4x400k", "882c5a1b569c1f3eb8bc90595ab0af124df3919d", "muon_slot4"),
        "805eb4d2620a56ae3c47c4c22e9696d7e59e6c429e4349d9e22a5bcceb2ebee2",
        tuple(range(50_000, 400_001, 50_000)),
    ),
    Cell(
        "muon-cbeta-slot5", "muon-cbeta-clock", "882c5a1b569c1f3eb8bc90595ab0af124df3919d",
        _config("l128-batch256-muon-cbeta-clock-4x400k", "882c5a1b569c1f3eb8bc90595ab0af124df3919d", "muon_slot5"),
        "4ccfa580bba6d07a8c7fa6072c0ae70959894159224eee041e1ff464c9ed4d63",
        tuple(range(50_000, 400_001, 50_000)),
    ),
    Cell(
        "muon-cbeta-slot5-ss3di", "muon-cbeta-clock", "882c5a1b569c1f3eb8bc90595ab0af124df3919d",
        _config("l128-batch256-muon-cbeta-clock-4x400k", "882c5a1b569c1f3eb8bc90595ab0af124df3919d", "muon_slot5_ss3di"),
        "3822f46a8935576de8c1086f4a14a5a136379a3d392f6eb9fb29a831a351c2a7",
        tuple(range(50_000, 400_001, 50_000)),
    ),
    Cell(
        "muon-slot4-ss3di-repair", "muon-slot4-ss3di-repair", "192357b3aeea2d29846edf11852115802e833187",
        _config("l128-batch256-muon-slot4-ss3di-repair-400k", "192357b3aeea2d29846edf11852115802e833187", "muon_slot4_ss3di"),
        "0dc4faf7bf497b6cc8b924652142a790f3a42e20b7e36b6409962f6d26a919f8",
        tuple(range(50_000, 400_001, 50_000)),
    ),
    Cell(
        "signal-canonical-per-head", "signal-residual-shell", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac",
        _config("signal-dit-residual-shell-b256-4x200k", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac", "canonical_per_head"),
        "7bcc1ccd54d8bd83b86c982c90f2834525f439b797392b4afcdac8456e862f3b",
        (50_000, 100_000, 150_000, 200_000),
    ),
    Cell(
        "signal-canonical-standard-qk", "signal-residual-shell", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac",
        _config("signal-dit-residual-shell-b256-4x200k", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac", "canonical_standard_qk"),
        "62d1bc757b696fb5980b939592df190582dcf302c7fceb62f8af8b705382ce2f",
        (50_000, 100_000, 150_000, 200_000),
    ),
    Cell(
        "signal-stabilized-per-head", "signal-residual-shell", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac",
        _config("signal-dit-residual-shell-b256-4x200k", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac", "stabilized_per_head"),
        "bc97bf2a571ffb050f4e17ad2ceccbc7db9d0b34fc1b39c9a1ae6ba49f8261be",
        (50_000, 100_000, 150_000, 200_000),
    ),
    Cell(
        "signal-stabilized-standard-qk", "signal-residual-shell", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac",
        _config("signal-dit-residual-shell-b256-4x200k", "a4692d9daaa74e3d5a03fb1f9c56f2dbe26ea6ac", "stabilized_standard_qk"),
        "55f913f898ad449f2fd35894fc517d6aaca04f42439d1c2b2189b78a5fa72f48",
        (50_000, 100_000, 150_000, 200_000),
    ),
)

# The initial submission completed the two clean-control cells. The remaining
# cells failed uniformly in sampler config parsing and are the only ones retried.
RETRY_CELLS = tuple(cell for cell in CELLS if cell.cell_id not in {"clean4458e63-adam", "clean4458e63-adamw"})


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
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
        for step in cell.milestones:
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


def build_workflow(code_commit: str, *, retry_failed: bool = False) -> tuple[PreparedWorkflow, dict[str, Any]]:
    short = code_commit[:7]
    cells = RETRY_CELLS if retry_failed else CELLS
    suffix = "retry" if retry_failed else "v1"
    workflow = f"hk-architecture-milestone-evaluation-50k-{short}-{suffix}"
    output_name = RETRY_OUTPUT_ROOT_NAME if retry_failed else OUTPUT_ROOT_NAME
    output_root = REMOTE_RUN_ROOT / output_name / code_commit / "v1"
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
        for cell in cells:
            tasks.extend(shared._evaluation_tasks(
                cell,
                workflow=workflow,
                code_commit=code_commit,
                output_root=output_root,
                cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{cell.source_commit[:7]}"),
                profiles=profiles,
                train_id=None,
                train_dir=cell.parent_config.parent,
                attestation_artifact=None,
            ))
    finally:
        shared._patches, shared.SAMPLE_SEED = old_patches, old_seed
    expected_tasks = sum(len(cell.milestones) for cell in cells) * 3
    if len(tasks) != expected_tasks:
        raise AssertionError(f"expected {expected_tasks} tasks, built {len(tasks)}")
    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )
    description = {
        "schema": "hierarchical-kaveh.architecture-milestone-evaluation.v1",
        "workflow_id": workflow,
        "request_id": prepared.request_id,
        "project_id": PROJECT_ID,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "milestones_submitted": sorted({step for cell in cells for step in cell.milestones}),
        "previously_evaluated_milestones": PREVIOUSLY_EVALUATED,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "family": cell.family,
                "source_commit": cell.source_commit,
                "config": str(cell.parent_config),
                "config_sha256": cell.config_sha256,
                "milestones": list(cell.milestones),
            }
            for cell in cells
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
        "task_counts": {
            "sampling": sum(len(cell.milestones) for cell in cells),
            "esmfold": sum(len(cell.milestones) for cell in cells),
            "progres_analysis": sum(len(cell.milestones) for cell in cells),
        },
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
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    workflow, bundle = build_workflow(code_commit, retry_failed=args.retry_failed)
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
