#!/usr/bin/env python3
"""Recover and analyze the one missing 100k decoder follow-up sample."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import load_environment_profile, prepare_run, submit_scruffy  # noqa: E402
import koochak  # noqa: E402

KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263106")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site")
PROJECT_ID = "kaveh-ce20-20260806"
SOURCE_COMMIT = "1ecf2c66b659d0f0b335405e231dda5e4e7a40ae"
SOURCE_ROOT = REMOTE_RUN_ROOT / "local-center-decoder-followup-8x100k" / SOURCE_COMMIT / "v1"
VARIANT = "coarse12_residue8_decoder_double_331286"
TRAIN_ROOT = SOURCE_ROOT / "train" / VARIANT
CHECKPOINT = TRAIN_ROOT / "step0100000.pt"
SAMPLE_DIR = SOURCE_ROOT / "samples" / "step100000-recovery" / VARIANT
ANALYSIS_DIR = SOURCE_ROOT / "analysis" / "step100000-recovery-launch"
ANALYSIS_OUTPUT = SOURCE_ROOT / "analysis" / "milestone_step100000-recovery.json"


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_checkout() -> str:
    expected_koochak = (REPO_ROOT / "external/koochak").resolve()
    unrelated = [
        line for line in _git("status", "--porcelain").splitlines()
        if line and not line.endswith(" external/koochak")
    ]
    if unrelated:
        raise RuntimeError(f"submission requires a clean campaign checkout: {unrelated}")
    if Path(koochak.__file__).resolve().is_relative_to(expected_koochak) is False:
        raise RuntimeError("loaded Koochak from the wrong checkout")
    if _git("rev-parse", "HEAD", cwd=expected_koochak) != KOOCHAK_COMMIT:
        raise RuntimeError("submission requires the pinned Koochak commit")
    commit = _git("rev-parse", "HEAD")
    expected_repo = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{commit[:7]}"
    if REPO_ROOT.resolve() != expected_repo:
        raise RuntimeError(f"run from the independent checkout {expected_repo}")
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size <= 0:
        raise FileNotFoundError(f"verified checkpoint is missing: {CHECKPOINT}")
    if not SCRUFFY_ROOT.exists() or not SCRUFFY_SITE.exists():
        raise RuntimeError("Scruffy launch paths are unavailable")
    return commit


def _prepare(commit: str) -> tuple[str, list[dict[str, object]]]:
    short = commit[:7]
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    gpu_profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml")
    cpu_profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-cpu.yaml")
    workflow_id = f"hk-local-center-decoder-followup-recovery-100k-{SOURCE_COMMIT[:7]}-v1"
    sample = prepare_run(
        name=f"hk-decoder-followup-sample-100000-recovery-{SOURCE_COMMIT[:7]}",
        profile=gpu_profile,
        python_args=[
            "{cwd}/scripts/sample_short128_milestone.py",
            "--config", str(TRAIN_ROOT / "config.yaml"),
            "--checkpoint", str(CHECKPOINT), "--output-dir", str(SAMPLE_DIR),
            "--lengths", "64,96,128", "--samples-per-length", "32",
            "--batch-size", "32", "--seed", "20260813", "--precision", "bf16",
            "--compile",
        ],
        cwd=str(remote_cwd), run_dir=str(SAMPLE_DIR), base_config=None,
    )
    analysis = prepare_run(
        name=f"hk-decoder-followup-analysis-100000-recovery-{SOURCE_COMMIT[:7]}",
        profile=cpu_profile,
        python_args=["{cwd}/scripts/analyze_local_center_decoder_followup_recovery.py"],
        cwd=str(remote_cwd), run_dir=str(ANALYSIS_DIR), base_config=None,
    )
    return workflow_id, [
        {"task_id": "sample-100000-recovery", "run": sample, "resource": "sample", "needs": []},
        {
            "task_id": "analysis-100000-recovery", "run": analysis, "resource": "analysis",
            "needs": [{"task_id": "sample-100000-recovery", "condition": "succeeded"}],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = _validate_checkout()
    workflow_id, tasks = _prepare(commit)
    if args.dry_run:
        result = [
            {"task_id": task["task_id"], "name": task["run"].name,
             "resource": task["resource"], "needs": task["needs"]}
            for task in tasks
        ]
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest  # noqa: PLC0415

        resources = {
            "sample": ResourceRequest(nodes=1, gpus_per_node=1, cpus_per_node=14, memory_gb_per_node=128, time_limit_seconds=172800),
            "analysis": ResourceRequest(nodes=1, gpus_per_node=1, cpus_per_node=2, memory_gb_per_node=16, time_limit_seconds=7200),
        }
        result = []
        for task in tasks:
            submitted = submit_scruffy(
                task["run"], root=SCRUFFY_ROOT, resources=resources[task["resource"]],
                request_id=f"{workflow_id}/{task['task_id']}/v1", project_id=PROJECT_ID,
                workflow_id=workflow_id, task_id=task["task_id"], needs=task["needs"],
            )
            result.append({"task_id": task["task_id"], **submitted})
    print(json.dumps({"workflow_id": workflow_id, "tasks": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
