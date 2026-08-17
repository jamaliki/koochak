#!/usr/bin/env python3
"""Submit a no-Kabsch coordinate-loss control and its fixed-panel diagnostics."""

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
    ConfigPatch,
    load_environment_profile,
    prepare_run,
    submit_scruffy,
)
import koochak  # noqa: E402


BASE_MAIN_COMMIT = "1e3746a"
KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site")
PROJECT_ID = "kaveh-ce20-20260806"
BASE_CONFIG = REPO_ROOT / "configs/experiments/local_center_secondary_structure_4x100k.yaml"
GPU_PROFILE = REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml"
CPU_PROFILE = REPO_ROOT / "environments/tokyo-pair-distogram-cpu.yaml"
MILESTONES = (50_000, 100_000)


@dataclass(frozen=True)
class Control:
    name: str = "baseline_no_kabsch"
    wandb_group: str = "local-center-coordinate-alignment-1x100k-v1"


CONTROL = Control()


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


def _patches(run_dir: Path) -> list[ConfigPatch]:
    return [
        ConfigPatch("loss.align_coordinate_loss", False),
        ConfigPatch("logging.csv_path", str(run_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "log.jsonl")),
        ConfigPatch("wandb.name", "local-center-coordinate-alignment-1x100k-baseline-no-kabsch"),
        ConfigPatch("wandb.group", CONTROL.wandb_group),
        ConfigPatch("wandb.tags", ["local-center-coordinate-alignment-1x100k", "no-kabsch"]),
    ]


def _assert_rendered_config(prepared, patches: list[ConfigPatch]) -> None:
    allowed = {
        "loss.align_coordinate_loss", "logging.csv_path", "logging.jsonl_path",
        "train.out_dir", "wandb.name", "wandb.group", "wandb.tags",
    }
    if any(patch.path not in allowed for patch in patches):
        raise AssertionError("launcher contains a patch outside the no-Kabsch contract")
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    actual = OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)
    expected = OmegaConf.load(BASE_CONFIG)
    OmegaConf.update(expected, "train.out_dir", prepared.run_dir, force_add=True)
    for patch in patches:
        OmegaConf.update(expected, patch.path, patch.value, merge=patch.merge, force_add=True)
    if actual != OmegaConf.to_container(expected, resolve=True):
        raise AssertionError(f"rendered config differs from allowed patches: {prepared.name}")


def _task(task_id: str, prepared, resource: str, needs=None) -> dict[str, object]:
    return {"task_id": task_id, "run": prepared, "resource": resource, "needs": needs or []}


def _prepare_tasks(commit: str) -> tuple[str, list[dict[str, object]]]:
    short = commit[:7]
    output_root = REMOTE_RUN_ROOT / "local-center-coordinate-alignment-1x100k" / commit
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    gpu_profile = load_environment_profile(GPU_PROFILE)
    cpu_profile = load_environment_profile(CPU_PROFILE)
    workflow_id = f"hk-local-center-coordinate-alignment-1x100k-{short}-v1"
    train_dir = output_root / "train" / CONTROL.name
    train = prepare_run(
        name=f"hk-local-center-align-train-{CONTROL.name}-{short}",
        profile=gpu_profile,
        python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}"],
        cwd=str(remote_cwd), run_dir=str(train_dir), base_config=BASE_CONFIG,
        patches=_patches(train_dir),
    )
    _assert_rendered_config(train, _patches(train_dir))
    tasks = [_task("train-baseline-no-kabsch", train, "train")]
    sample_ids: list[str] = []
    for step in MILESTONES:
        sample_dir = output_root / "samples" / f"step{step:06d}" / CONTROL.name
        checkpoint = train_dir / f"step{step:07d}.pt"
        sample = prepare_run(
            name=f"hk-local-center-align-sample-{step}-{CONTROL.name}-{short}",
            profile=gpu_profile,
            python_args=[
                "{cwd}/scripts/sample_short128_milestone.py",
                "--config", str(train_dir / "config.yaml"),
                "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
                "--lengths", "64,96,128", "--samples-per-length", "32",
                "--batch-size", "32", "--seed", "20260817", "--precision", "bf16",
                "--compile",
            ],
            cwd=str(remote_cwd), run_dir=str(sample_dir), base_config=None,
        )
        task_id = f"sample-{step}-baseline-no-kabsch"
        sample_ids.append(task_id)
        tasks.append(_task(task_id, sample, "sample", [
            {"task_id": "train-baseline-no-kabsch", "condition": "succeeded"},
        ]))

    for step, sample_id in zip(MILESTONES, sample_ids, strict=True):
        analysis_dir = output_root / "analysis" / f"step{step:06d}"
        analysis = prepare_run(
            name=f"hk-local-center-align-analysis-{step}-{short}",
            profile=cpu_profile,
            python_args=[
                "{cwd}/scripts/analyze_sample_panel.py",
                str(output_root / "samples" / f"step{step:06d}"),
                "--output", str(output_root / "analysis" / f"step{step:06d}.json"),
            ],
            cwd=str(remote_cwd), run_dir=str(analysis_dir), base_config=None,
        )
        tasks.append(_task(f"analysis-{step}", analysis, "cpu", [
            {"task_id": sample_id, "condition": "succeeded"},
        ]))
    return workflow_id, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = _validate_checkout()
    workflow_id, tasks = _prepare_tasks(commit)
    if args.dry_run:
        result = [
            {"task_id": item["task_id"], "resource": item["resource"],
             "run_dir": item["run"].run_dir, "needs": item["needs"]}
            for item in tasks
        ]
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest  # noqa: PLC0415

        resource_values = {
            "train": (1, 14, 240, 172_800),
            "sample": (1, 14, 128, 172_800),
            "cpu": (1, 2, 16, 7_200),
        }
        result = []
        for item in tasks:
            gpus, cpus, memory, seconds = resource_values[item["resource"]]
            submitted = submit_scruffy(
                item["run"], root=SCRUFFY_ROOT,
                resources=ResourceRequest(
                    nodes=1, gpus_per_node=gpus, cpus_per_node=cpus,
                    memory_gb_per_node=memory, time_limit_seconds=seconds,
                ),
                request_id=f"{workflow_id}/{item['task_id']}/v1", project_id=PROJECT_ID,
                workflow_id=workflow_id, task_id=item["task_id"], needs=item["needs"],
            )
            result.append({"task_id": item["task_id"], **submitted})
    print(json.dumps({"workflow_id": workflow_id, "task_count": len(tasks), "tasks": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
