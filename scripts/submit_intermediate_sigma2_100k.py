#!/usr/bin/env python3
"""Submit intermediate distogram feedback training with a 2-sigma CE gate."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import ConfigPatch, load_environment_profile, prepare_run, submit_scruffy  # noqa: E402
import koochak  # noqa: E402

BASE_MAIN_COMMIT = "114094b"
KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-614e355/site")
PROJECT_ID = "kaveh-ce20-20260806"
BASE_CONFIG = REPO_ROOT / "configs/experiments/intermediate_local_center_distogram_100k.yaml"
GPU_PROFILE = REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml"
MILESTONES = (10_000, 25_000, 50_000, 100_000)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def _validate() -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean checkout")
    if _git("rev-parse", "HEAD", cwd=REPO_ROOT / "external/koochak") != KOOCHAK_COMMIT:
        raise RuntimeError("submission requires the pinned Koochak commit")
    subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", BASE_MAIN_COMMIT, "HEAD"], check=True)
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from independent checkout {expected}")
    if not SCRUFFY_ROOT.exists() or not SCRUFFY_SITE.exists():
        raise RuntimeError("Scruffy launch paths are unavailable")
    if not Path(koochak.__file__).resolve().is_relative_to((REPO_ROOT / "external/koochak").resolve()):
        raise RuntimeError("loaded Koochak from the wrong checkout")
    return _git("rev-parse", "HEAD")


def _checkpoint(train_dir: Path, step: int) -> Path:
    width = 7 if step == 100_000 else 9
    return train_dir / f"step{step:0{width}d}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = _validate()
    short = commit[:7]
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    output_root = REMOTE_RUN_ROOT / "geometry-distogram-100k-sigma2" / short
    train_dir = output_root / "train" / "intermediate_local_center_sigma2"
    workflow_id = f"hk-geometry-intermediate-sigma2-100k-{short}-v1"
    profile = load_environment_profile(GPU_PROFILE)
    patches = [
        ConfigPatch("loss.aatype_sigma_max", 2.0),
        ConfigPatch("logging.csv_path", str(train_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(train_dir / "log.jsonl")),
        ConfigPatch("train.ckpt_every", 5_000),
        ConfigPatch("wandb.name", "geometry-intermediate-sigma2-100k"),
        ConfigPatch("wandb.group", "current-main-intermediate-sigma2-v1"),
    ]
    train = prepare_run(
        name=f"hk-geometry-intermediate-sigma2-train-{short}",
        profile=profile,
        python_args=["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1", "-m", "hierarchical_kaveh.train", "--config", "{config}"],
        cwd=str(remote_cwd), run_dir=str(train_dir), base_config=BASE_CONFIG, patches=patches,
    )
    tasks = [("train", train, 1, 14, 240, 172800, [])]
    for step in MILESTONES:
        sample_dir = output_root / "samples" / f"step{step:06d}" / "intermediate_local_center_sigma2"
        sample = prepare_run(
            name=f"hk-geometry-intermediate-sigma2-sample-{step}-{short}",
            profile=profile,
            python_args=[
                "{cwd}/scripts/wait_for_checkpoint_and_sample.py", "--checkpoint", str(_checkpoint(train_dir, step)),
                "--timeout-seconds", "172800", "{cwd}/scripts/sample_short128_milestone.py",
                "--config", str(train_dir / "config.yaml"), "--checkpoint", str(_checkpoint(train_dir, step)),
                "--output-dir", str(sample_dir), "--lengths", "64,96,128", "--samples-per-length", "32",
                "--batch-size", "32", "--seed", "20260813", "--precision", "bf16", "--compile",
            ],
            cwd=str(remote_cwd), run_dir=str(sample_dir), base_config=None,
        )
        tasks.append((f"sample-{step}", sample, 1, 14, 128, 172800, []))
    if args.dry_run:
        result = [{"task_id": task_id, "name": run.name, "run_dir": run.run_dir} for task_id, run, *_ in tasks]
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest  # noqa: PLC0415
        result = []
        for task_id, run, gpus, cpus, memory, seconds, needs in tasks:
            submitted = submit_scruffy(
                run, root=SCRUFFY_ROOT,
                resources=ResourceRequest(nodes=1, gpus_per_node=gpus, cpus_per_node=cpus, memory_gb_per_node=memory, time_limit_seconds=seconds),
                request_id=f"{workflow_id}/{task_id}/v1", project_id=PROJECT_ID,
                workflow_id=workflow_id, task_id=task_id, needs=needs,
            )
            result.append({"task_id": task_id, **submitted})
    print(json.dumps({"workflow_id": workflow_id, "tasks": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
