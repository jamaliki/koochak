#!/usr/bin/env python3
"""Recover only the collapse summary from checksum-verified completed analyses."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import PreparedTask, PreparedWorkflow, load_environment_profile, submit_scruffy_workflow

from scripts.submit_patch_coarse_factorial import (
    KOOCHAK_COMMIT,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_ROOT,
    _git,
)
from scripts.submit_patch_coarse_factorial_followup import SCRUFFY_COMMIT, SCRUFFY_SITE
from scripts.submit_patch_coarse_step_scale_sweep import _output, _stage_run
from scripts.submit_progres_collapse_audit import FOLLOWUP_ROOT, PANEL_COUNT


SOURCE_COMMIT = "62d73dd748c69535d8b2472d17c56a30e8b8ee9d"
SOURCE_ROOT = REMOTE_RUN_ROOT / "progres-training-collapse-L128" / SOURCE_COMMIT
PROFILE = REPO_ROOT / "environments" / "tokyo-collapse-progres-cpu.yaml"
RUNNER = "{cwd}/scripts/run_with_kaveh_python.py"
RESOURCES = {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 2, "memory_gb_per_node": 8, "time_limit_seconds": 1_800}


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-progres-training-collapse-recovery-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "progres-training-collapse-recovery-L128" / code_commit
    output = output_root / "collapse_summary.json"
    artifact = _output(
        "analysis/collapse-summary.json", output,
        stage="analysis", workflow=workflow, task="summarize", kind="file", expected_records=1,
    )
    run = _stage_run(
        stage="analysis", task="summarize", workflow=workflow, artifact=artifact,
        run_dir=output_root / "summarize.managed", profile=load_environment_profile(PROFILE),
        command=[
            RUNNER, "{cwd}/scripts/summarize_progres_collapse.py",
            "--analysis-root", str(SOURCE_ROOT / "history"),
            "--training-root", str(FOLLOWUP_ROOT / "train" / "L128"),
            "--training-data-progres", str(SOURCE_ROOT / "training_data" / "progres_diversity.json"),
            "--expected-count", str(PANEL_COUNT), "--output", str(output),
        ],
    )
    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=(PreparedTask("summarize", run, RESOURCES, recovery=RECOVERY),),
    )


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from the independent checkout {expected}")
    if _git("rev-parse", "HEAD", cwd=REPO_ROOT / "external" / "koochak") != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    panels = sorted((SOURCE_ROOT / "history").glob("step*/*/progres_diversity.json"))
    expected_files = panels + [SOURCE_ROOT / "training_data" / "progres_diversity.json"]
    missing = [str(file) for file in expected_files for file in (file, file.with_name(file.name + ".ready.json")) if not file.is_file()]
    if len(panels) != PANEL_COUNT or missing:
        raise RuntimeError(f"recovery inputs incomplete: panels={len(panels)}/{PANEL_COUNT}, missing={missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    workflow = build_workflow(code_commit)
    description = {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "code_commit": code_commit,
        "source_commit": SOURCE_COMMIT,
        "source_panel_count": PANEL_COUNT,
        "task_count": 1,
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True))
        return
    _validate_online(code_commit)
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status
    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    result = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "submission": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
