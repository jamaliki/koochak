#!/usr/bin/env python3
"""Recover one failed 200k coarse-continuation sampler and analyze the panel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    load_environment_profile,
    prepare_run,
    submit_scruffy,
)
import koochak  # noqa: E402


BASE_MAIN_COMMIT = "1fc3c39"
KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site")
PROJECT_ID = "kaveh-ce20-20260806"
SOURCE_COMMIT = "62233ea12be98e9220ebf532c0422fcb58d4b521"
RUN_ROOT = REMOTE_RUN_ROOT / "local-center-coarse-continuation-6x200k" / SOURCE_COMMIT / "v1"
OUTPUT_ROOT = RUN_ROOT / "manual-sampling-200k-v2"
TRAIN_ROOT = RUN_ROOT / "train"
MILESTONE = 200_000
RECOVERY_VARIANT = "coarse_deeper16_intermediate_331633"
ALL_VARIANTS = (
    "coarse_deep_331233",
    "coarse_deep_intermediate_331233",
    "coarse_deeper16_331633",
    "coarse_deeper16_intermediate_331633",
    "coarse_deeper20_332033",
    "coarse_deeper20_intermediate_332033",
)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_checkout() -> str:
    expected_koochak = (REPO_ROOT / "external/koochak").resolve()
    status = [line for line in _git("status", "--porcelain").splitlines() if line]
    unrelated = [line for line in status if not line.endswith(" external/koochak")]
    if unrelated:
        raise RuntimeError(f"submission requires a clean campaign checkout: {unrelated}")
    if not Path(koochak.__file__).resolve().is_relative_to(expected_koochak):
        raise RuntimeError("loaded Koochak from the wrong checkout")
    if _git("rev-parse", "HEAD", cwd=expected_koochak) != KOOCHAK_COMMIT:
        raise RuntimeError("submission requires the pinned Koochak commit")
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", BASE_MAIN_COMMIT, "HEAD"],
        check=True,
    )
    commit = _git("rev-parse", "HEAD")
    expected_repo = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{commit[:7]}"
    if REPO_ROOT.resolve() != expected_repo:
        raise RuntimeError(f"run from the independent checkout {expected_repo}")
    if not SCRUFFY_ROOT.exists() or not SCRUFFY_SITE.exists():
        raise RuntimeError("Scruffy launch paths are unavailable")
    for variant in ALL_VARIANTS:
        if variant == RECOVERY_VARIANT:
            continue
        manifest = OUTPUT_ROOT / "samples" / f"step{MILESTONE:06d}" / variant / "manifest.json"
        if not manifest.is_file() or manifest.stat().st_size <= 0:
            raise FileNotFoundError(f"completed sample manifest is missing: {manifest}")
    return commit


def _checkpoint() -> Path:
    directory = TRAIN_ROOT / RECOVERY_VARIANT
    for filename in ("step0200000.pt", "step000200000.pt"):
        checkpoint = directory / filename
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return checkpoint
    raise FileNotFoundError(f"verified 200k checkpoint is missing for {RECOVERY_VARIANT}")


def _prepare_tasks(commit: str) -> tuple[str, list[dict[str, object]]]:
    short = commit[:7]
    remote_cwd = REMOTE_CODE_ROOT / "hierarchical_kaveh_62233ea"
    sample_dir = OUTPUT_ROOT / "samples" / f"step{MILESTONE:06d}" / RECOVERY_VARIANT
    recovery_root = RUN_ROOT / "manual-sampling-200k-v2-recovery"
    gpu_profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml")
    cpu_profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-cpu.yaml")
    workflow_id = f"hk-local-center-coarse-continuation-sample-200k-recovery-{short}-v1"
    checkpoint = _checkpoint()
    sample = prepare_run(
        name=f"hk-coarse-continuation-manual-sample-200000-recovery-{RECOVERY_VARIANT}-{short}",
        profile=gpu_profile,
        python_args=[
            "{cwd}/scripts/sample_short128_milestone.py",
            "--config", str(TRAIN_ROOT / RECOVERY_VARIANT / "config.yaml"),
            "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
            "--lengths", "64,96,128", "--samples-per-length", "32",
            "--batch-size", "32", "--seed", "20260817", "--precision", "bf16",
            "--compile",
        ],
        cwd=str(remote_cwd), run_dir=str(recovery_root / "sample"), base_config=None,
    )
    analysis_output = OUTPUT_ROOT / "analysis" / "milestone_step200000-recovery.json"
    analysis = prepare_run(
        name=f"hk-coarse-continuation-manual-analysis-200000-recovery-{short}",
        profile=cpu_profile,
        python_args=[
            "{cwd}/scripts/analyze_sample_panel.py",
            str(OUTPUT_ROOT / "samples" / f"step{MILESTONE:06d}"),
            "--output", str(analysis_output),
        ],
        cwd=str(remote_cwd), run_dir=str(recovery_root / "analysis"), base_config=None,
    )
    return workflow_id, [
        {"task_id": "sample-recovery", "run": sample, "resource": "sample", "needs": []},
        {
            "task_id": "analysis-recovery", "run": analysis, "resource": "cpu",
            "needs": [{"task_id": "sample-recovery", "condition": "succeeded"}],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = _validate_checkout()
    workflow_id, tasks = _prepare_tasks(commit)
    if args.dry_run:
        result = [
            {"task_id": item["task_id"], "resource": item["resource"], "needs": item["needs"]}
            for item in tasks
        ]
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest  # noqa: PLC0415

        resources = {
            "sample": ResourceRequest(nodes=1, gpus_per_node=1, cpus_per_node=14, memory_gb_per_node=128, time_limit_seconds=43_200),
            "cpu": ResourceRequest(nodes=1, gpus_per_node=0, cpus_per_node=2, memory_gb_per_node=16, time_limit_seconds=7_200),
        }
        result = []
        for item in tasks:
            submitted = submit_scruffy(
                item["run"], root=SCRUFFY_ROOT, resources=resources[item["resource"]],
                request_id=f"{workflow_id}/{item['task_id']}/v1", project_id=PROJECT_ID,
                workflow_id=workflow_id, task_id=item["task_id"], needs=item["needs"],
            )
            result.append({"task_id": item["task_id"], **submitted})
    print(json.dumps({"workflow_id": workflow_id, "task_count": len(tasks), "tasks": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
