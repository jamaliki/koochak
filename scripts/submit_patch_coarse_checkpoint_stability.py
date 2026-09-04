#!/usr/bin/env python3
"""Submit peak-versus-late checkpoint stability probes through Koochak/Scruffy."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import PreparedTask, PreparedWorkflow, submit_scruffy_workflow  # noqa: E402
from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    GPU_PROFILE,
    KOOCHAK_COMMIT,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_ROOT,
    _git,
)
from scripts.submit_patch_coarse_factorial_followup import (  # noqa: E402
    SCRUFFY_COMMIT,
    SCRUFFY_SITE,
    _load_profile,
)
from scripts.submit_patch_coarse_late_step_scale import FOLLOWUP_ROOT, SOURCE_CELLS  # noqa: E402
from scripts.submit_patch_coarse_step_scale_sweep import RESOURCES, _output, _stage_run  # noqa: E402


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-checkpoint-stability-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-checkpoint-stability-L128" / code_commit
    profile = _load_profile(GPU_PROFILE)
    tasks = []
    for source in SOURCE_CELLS:
        task_id = f"probe-{source.cell_id}"
        output = output_root / source.cell_id / "stability.json"
        artifact = _output(
            f"analysis/{source.cell_id}/stability.json",
            output,
            stage="analysis",
            workflow=workflow,
            task=task_id,
            kind="file",
            expected_records=1,
        )
        run = _stage_run(
            stage="analysis",
            task=task_id,
            workflow=workflow,
            artifact=artifact,
            run_dir=output.parent.with_name(output.parent.name + ".managed"),
            profile=profile,
            command=[
                "{cwd}/scripts/diagnose_checkpoint_stability.py",
                "--config", str(source.train_dir / "config.yaml"),
                "--checkpoint", str(source.train_dir / f"step{source.step:09d}.pt"),
                "--output", str(output),
                "--length", "128",
                "--precision", "bf16",
            ],
        )
        tasks.append(PreparedTask(task_id, run, RESOURCES["sample"], recovery=RECOVERY))
    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    koochak_root = (REPO_ROOT / "external" / "koochak").resolve()
    if _git("rev-parse", "HEAD", cwd=koochak_root) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from the independent checkout {expected}")
    missing = [
        str(file)
        for source in SOURCE_CELLS
        for file in (
            source.train_dir / "config.yaml",
            source.train_dir / f"step{source.step:09d}.pt",
        )
        if not file.is_file()
    ]
    if missing:
        raise RuntimeError(f"probe inputs are missing: {missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _validate_online(code_commit)
    workflow = build_workflow(code_commit)
    description = {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "code_commit": code_commit,
        "source_commit": str(FOLLOWUP_ROOT.name),
        "source_cells": [source.cell_id for source in SOURCE_CELLS],
        "task_count": len(workflow.tasks),
        "task_states": Counter("gpu" if task.resources["gpus_per_node"] else "cpu" for task in workflow.tasks),
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True))
        return
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415
    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "submission": submission}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
