#!/usr/bin/env python3
"""Sample verified 150k checkpoints from the six-arm coarse continuation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    load_environment_profile,
    prepare_run,
    submit_scruffy,
)
import koochak  # noqa: E402


BASE_MAIN_COMMIT = "88306f9"
KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site")
PROJECT_ID = "kaveh-ce20-20260806"
SOURCE_COMMIT = "62233ea12be98e9220ebf532c0422fcb58d4b521"
TRAIN_ROOT = REMOTE_RUN_ROOT / "local-center-coarse-continuation-6x200k" / SOURCE_COMMIT / "v1" / "train"
MILESTONE = 150_000


@dataclass(frozen=True)
class Variant:
    name: str


VARIANTS = (
    Variant("coarse_deep_331233"),
    Variant("coarse_deep_intermediate_331233"),
    Variant("coarse_deeper16_331633"),
    Variant("coarse_deeper16_intermediate_331633"),
    Variant("coarse_deeper20_332033"),
    Variant("coarse_deeper20_intermediate_332033"),
)
CONTROL = "coarse_deep_331233"


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
    return commit


def _checkpoint(variant: Variant) -> Path:
    checkpoint = TRAIN_ROOT / variant.name / "step000150000.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"verified 150k checkpoint is missing: {checkpoint}")
    return checkpoint


def _prepare_tasks(commit: str) -> tuple[str, list[dict[str, object]]]:
    short = commit[:7]
    remote_cwd = REMOTE_CODE_ROOT / "hierarchical_kaveh_62233ea"
    output_root = REMOTE_RUN_ROOT / "local-center-coarse-continuation-6x200k" / SOURCE_COMMIT / "v1" / "manual-sampling-150k"
    gpu_profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml")
    cpu_profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-cpu.yaml")
    workflow_id = f"hk-local-center-coarse-continuation-sample-150k-{short}-v1"
    tasks: list[dict[str, object]] = []
    sample_ids: list[str] = []
    for variant in VARIANTS:
        checkpoint = _checkpoint(variant)
        sample_dir = output_root / "samples" / f"step{MILESTONE:06d}" / variant.name
        prepared = prepare_run(
            name=f"hk-coarse-continuation-manual-sample-150000-{variant.name}-{short}",
            profile=gpu_profile,
            python_args=[
                "{cwd}/scripts/sample_short128_milestone.py",
                "--config", str(TRAIN_ROOT / variant.name / "config.yaml"),
                "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
                "--lengths", "64,96,128", "--samples-per-length", "32",
                "--batch-size", "32", "--seed", "20260817", "--precision", "bf16",
                "--compile",
            ],
            cwd=str(remote_cwd), run_dir=str(sample_dir), base_config=None,
        )
        task_id = f"sample-150000-{variant.name}"
        sample_ids.append(task_id)
        tasks.append({"task_id": task_id, "run": prepared, "resource": "sample", "needs": []})

    analysis_dir = output_root / "analysis" / f"step{MILESTONE:06d}"
    analysis_output = output_root / "analysis" / f"milestone_step{MILESTONE:06d}.json"
    analysis = prepare_run(
        name=f"hk-coarse-continuation-manual-analysis-150000-{short}",
        profile=cpu_profile,
        python_args=[
            "{cwd}/scripts/analyze_sample_panel.py",
            str(output_root / "samples" / f"step{MILESTONE:06d}"),
            "--output", str(analysis_output),
        ],
        cwd=str(remote_cwd), run_dir=str(analysis_dir), base_config=None,
    )
    tasks.append({
        "task_id": "analysis-150000", "run": analysis, "resource": "cpu",
        "needs": [{"task_id": task_id, "condition": "succeeded"} for task_id in sample_ids],
    })
    return workflow_id, tasks


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
