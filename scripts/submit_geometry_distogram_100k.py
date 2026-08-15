#!/usr/bin/env python3
"""Submit the current-main geometry/distogram 100k campaign through Scruffy."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    load_environment_profile,
    prepare_run,
    stage_run,
    submit_scruffy,
)
import koochak  # noqa: E402


BASE_MAIN_COMMIT = "114094b"
KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-614e355/site")
PROJECT_ID = "kaveh-ce20-20260806"
BASE_CONFIG = REPO_ROOT / "configs/experiments/local_center_distogram_100k.yaml"
GPU_PROFILE = REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml"
CPU_PROFILE = REPO_ROOT / "environments/tokyo-pair-distogram-cpu.yaml"
MILESTONES = (10_000, 25_000, 50_000, 100_000)
EARLY_SAMPLE_MILESTONES = frozenset({25_000})
CANARY_ARMS = ("baseline", "local_center", "intermediate_local_center")
TRAIN_ARMS = ("local_center", "intermediate_local_center")


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_checkout() -> str:
    expected_koochak = (REPO_ROOT / "external/koochak").resolve()
    if not Path(koochak.__file__).resolve().is_relative_to(expected_koochak):
        raise RuntimeError("loaded Koochak from the wrong checkout")
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
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
    missing = [str(item) for item in (SCRUFFY_ROOT, SCRUFFY_SITE) if not item.exists()]
    if missing:
        raise RuntimeError(f"required launch paths are missing: {missing}")
    return commit


def _arm_patches(arm: str, run_dir: Path, *, production: bool) -> list[ConfigPatch]:
    if arm == "baseline":
        geometry_mode, sc_geometry, intermediate, feedback, weight = "legacy", False, False, False, 0.0
    elif arm == "local_center":
        geometry_mode, sc_geometry, intermediate, feedback, weight = "local_center", True, False, False, 0.0
    elif arm == "intermediate_local_center":
        geometry_mode, sc_geometry, intermediate, feedback, weight = "local_center", True, True, True, 0.25
    else:
        raise ValueError(f"unknown arm {arm}")
    tags = [
        "current-main",
        "geometry-distogram-100k",
        arm,
        "lr=0.0003",
        "lddt=1",
        "terminal-distogram=0.5",
        f"intermediate-distogram={weight:g}",
        f"geometry={geometry_mode}",
        f"sc-geometry={str(sc_geometry).lower()}",
    ]
    patches = [
        ConfigPatch("model.pair_geometry_mode", geometry_mode),
        ConfigPatch("model.pair_self_conditioned_geometry", sc_geometry),
        ConfigPatch("model.intermediate_distograms", intermediate),
        ConfigPatch("model.intermediate_distogram_feedback", feedback),
        ConfigPatch("loss.intermediate_distogram_weight", weight),
        ConfigPatch("logging.csv_path", str(run_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "log.jsonl")),
        ConfigPatch("wandb.name", f"geometry-distogram-100k-{arm}"),
        ConfigPatch("wandb.tags", tags),
    ]
    if not production:
        patches.extend([
            ConfigPatch("wandb.enabled", False),
            ConfigPatch("wandb.mode", "disabled"),
            ConfigPatch("wandb.resume", None),
        ])
    return patches


def _prepare_tasks(commit: str) -> tuple[str, list[dict[str, object]]]:
    short = commit[:7]
    output_root = REMOTE_RUN_ROOT / "geometry-distogram-100k" / short
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    gpu_profile = load_environment_profile(GPU_PROFILE)
    cpu_profile = load_environment_profile(CPU_PROFILE)
    workflow_id = f"hk-geometry-distogram-100k-{short}-v1"
    tasks: list[dict[str, object]] = []

    for arm in CANARY_ARMS:
        run_dir = output_root / "canary" / arm
        prepared = prepare_run(
            name=f"hk-geometry-dist-canary-{arm}-{short}",
            profile=gpu_profile,
            python_args=[
                "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                "{cwd}/scripts/profile_short128_latency.py",
                "--base-config", "{config}",
                "--run-dir", "{run_dir}/profile",
                "--steps", "250",
                "--warmup", "75",
                "--summary-output", "{run_dir}/latency.json",
            ],
            cwd=str(remote_cwd),
            run_dir=str(run_dir),
            base_config=BASE_CONFIG,
            patches=_arm_patches(arm, run_dir, production=False),
        )
        tasks.append(dict(task_id=f"canary-{arm}", run=prepared, resource="canary", needs=[]))

    gate_dir = output_root / "throughput-guard"
    gate_args = [
        "{cwd}/scripts/check_throughput_guard.py",
        "--baseline", str(output_root / "canary/baseline/latency.json"),
        "--output", str(gate_dir / "report.json"),
        "--minimum-rows", "150",
        "--max-p50-regression", "0.08",
        "--max-p95-regression", "0.12",
    ]
    for arm in CANARY_ARMS:
        if arm != "baseline":
            gate_args.extend(["--candidate", str(output_root / f"canary/{arm}/latency.json")])
    gate = prepare_run(
        name=f"hk-geometry-dist-throughput-guard-{short}",
        profile=cpu_profile,
        python_args=gate_args,
        cwd=str(remote_cwd),
        run_dir=str(gate_dir),
        base_config=None,
    )
    tasks.append(dict(
        task_id="throughput-guard",
        run=gate,
        resource="cpu",
        needs=[{"task_id": f"canary-{arm}", "condition": "succeeded"} for arm in CANARY_ARMS],
    ))

    train_dirs: dict[str, Path] = {}
    for arm in TRAIN_ARMS:
        run_dir = output_root / "train" / arm
        train_dirs[arm] = run_dir
        prepared = prepare_run(
            name=f"hk-geometry-dist-train-{arm}-{short}",
            profile=gpu_profile,
            python_args=[
                "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                "-m", "hierarchical_kaveh.train", "--config", "{config}",
            ],
            cwd=str(remote_cwd),
            run_dir=str(run_dir),
            base_config=BASE_CONFIG,
            patches=_arm_patches(arm, run_dir, production=True),
        )
        tasks.append(dict(
            task_id=f"train-{arm}", run=prepared, resource="train",
            needs=[{"task_id": "throughput-guard", "condition": "succeeded"}],
        ))

    for step in MILESTONES:
        for arm in TRAIN_ARMS:
            task_id = f"sample-{step}-{arm}"
            run_dir = output_root / "samples" / f"step{step:06d}" / arm
            prepared = prepare_run(
                name=f"hk-geometry-dist-sample-{step}-{arm}-{short}",
                profile=gpu_profile,
                python_args=[
                    *(["{cwd}/scripts/wait_for_checkpoint_and_sample.py", "--checkpoint",
                       str(train_dirs[arm] / f"step{step:09d}.pt"), "--timeout-seconds", "172800",
                       "{cwd}/scripts/sample_short128_milestone.py"]
                      if step in EARLY_SAMPLE_MILESTONES else
                      ["{cwd}/scripts/sample_short128_milestone.py"]),
                    "--config", str(train_dirs[arm] / "config.yaml"),
                    "--checkpoint", str(train_dirs[arm] / f"step{step:09d}.pt"),
                    "--output-dir", str(run_dir),
                    "--lengths", "64,96,128",
                    "--samples-per-length", "32",
                    "--batch-size", "32",
                    "--seed", "20260813",
                    "--precision", "bf16",
                    "--compile",
                ],
                cwd=str(remote_cwd), run_dir=str(run_dir), base_config=None,
            )
            tasks.append(dict(
                task_id=task_id, run=prepared, resource="sample",
                needs=[] if step in EARLY_SAMPLE_MILESTONES else
                [{"task_id": f"train-{arm}", "condition": "succeeded"}],
            ))
    return workflow_id, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    commit = _validate_checkout()
    workflow_id, tasks = _prepare_tasks(commit)
    if args.dry_run:
        result = [
            {
                "task_id": item["task_id"], "name": item["run"].name,
                "run_dir": item["run"].run_dir, "resource": item["resource"],
                "needs": item["needs"],
            }
            for item in tasks
        ]
    elif args.stage_only:
        for item in tasks:
            stage_run(item["run"])
        result = [{"task_id": item["task_id"], "staged": True} for item in tasks]
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest

        resource_values = {
            "canary": (1, 14, 240, 7_200),
            "train": (1, 14, 240, 172_800),
            "sample": (1, 14, 128, 43_200),
            "cpu": (0, 2, 16, 1_800),
        }
        result = []
        for item in tasks:
            gpus, cpus, memory, seconds = resource_values[item["resource"]]
            resources = ResourceRequest(
                nodes=1, gpus_per_node=gpus, cpus_per_node=cpus,
                memory_gb_per_node=memory, time_limit_seconds=seconds,
            )
            submitted = submit_scruffy(
                item["run"], root=SCRUFFY_ROOT, resources=resources,
                request_id=f"{workflow_id}/{item['task_id']}/v1",
                project_id=PROJECT_ID, workflow_id=workflow_id,
                task_id=item["task_id"], needs=item["needs"],
            )
            result.append({"task_id": item["task_id"], **submitted})
    print(json.dumps({"workflow_id": workflow_id, "tasks": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
