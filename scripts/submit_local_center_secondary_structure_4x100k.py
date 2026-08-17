#!/usr/bin/env python3
"""Submit the controlled four-arm secondary-structure conditioning panel."""

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


BASE_MAIN_COMMIT = "8b67f48"
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
class SSVariant:
    name: str
    conditioning: bool
    prediction: bool
    recycling: bool
    data: bool
    loss_weight: float
    alpha: float = 0.5


VARIANTS = (
    SSVariant("baseline", False, False, False, False, 0.0),
    SSVariant("ss_input", True, False, False, True, 0.0),
    SSVariant("ss_aux", True, True, False, True, 0.1),
    SSVariant("ss_recurrent", True, True, True, True, 0.1),
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
    return commit


def _patches(variant: SSVariant, run_dir: Path, *, production: bool) -> list[ConfigPatch]:
    patches = [
        ConfigPatch("model.secondary_structure_conditioning", variant.conditioning),
        ConfigPatch("model.secondary_structure_prediction", variant.prediction),
        ConfigPatch("model.secondary_structure_self_conditioning", variant.recycling),
        ConfigPatch("model.secondary_structure_self_conditioning_alpha", variant.alpha),
        ConfigPatch("data.secondary_structure", variant.data),
        ConfigPatch("loss.secondary_structure_weight", variant.loss_weight),
        ConfigPatch("logging.csv_path", str(run_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "log.jsonl")),
        ConfigPatch("wandb.name", f"local-center-ss-4x100k-{variant.name}"),
        ConfigPatch("wandb.tags", ["local-center-ss-4x100k", variant.name]),
    ]
    if not production:
        patches.extend([
            ConfigPatch("wandb.enabled", False),
            ConfigPatch("wandb.mode", "disabled"),
            ConfigPatch("wandb.resume", None),
        ])
    return patches


def _assert_rendered_config(prepared, patches: list[ConfigPatch]) -> None:
    allowed = {
        "model.secondary_structure_conditioning", "model.secondary_structure_prediction",
        "model.secondary_structure_self_conditioning",
        "model.secondary_structure_self_conditioning_alpha", "data.secondary_structure",
        "loss.secondary_structure_weight", "logging.csv_path", "logging.jsonl_path",
        "train.out_dir", "wandb.name", "wandb.tags", "wandb.enabled", "wandb.mode",
        "wandb.resume",
    }
    if any(patch.path not in allowed for patch in patches):
        raise AssertionError("launcher contains a patch outside the panel contract")
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
    output_root = REMOTE_RUN_ROOT / "local-center-secondary-structure-4x100k" / commit
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    gpu_profile = load_environment_profile(GPU_PROFILE)
    cpu_profile = load_environment_profile(CPU_PROFILE)
    workflow_id = f"hk-local-center-secondary-structure-4x100k-{short}-v2"
    tasks: list[dict[str, object]] = []
    train_dirs: dict[str, Path] = {}
    train_ids: dict[str, str] = {}
    for variant in VARIANTS:
        run_dir = output_root / "train" / variant.name
        train_dirs[variant.name] = run_dir
        task_id = f"train-{variant.name}"
        train_ids[variant.name] = task_id
        prepared = prepare_run(
            name=f"hk-local-center-ss-train-{variant.name}-{short}",
            profile=gpu_profile,
            python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}"],
            cwd=str(remote_cwd), run_dir=str(run_dir), base_config=BASE_CONFIG,
            patches=_patches(variant, run_dir, production=True),
        )
        _assert_rendered_config(prepared, _patches(variant, run_dir, production=True))
        tasks.append(_task(task_id, prepared, "train"))

    sample_ids = {step: [] for step in MILESTONES}
    for step in MILESTONES:
        for variant in VARIANTS:
            train_dir = train_dirs[variant.name]
            sample_dir = output_root / "samples" / f"step{step:06d}" / variant.name
            checkpoint = train_dir / f"step{step:07d}.pt"
            prepared = prepare_run(
                name=f"hk-local-center-ss-sample-{step}-{variant.name}-{short}",
                profile=gpu_profile,
                python_args=[
                    "{cwd}/scripts/sample_short128_milestone.py",
                    "--config", str(train_dir / "config.yaml"),
                    "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
                    "--lengths", "64,96,128", "--samples-per-length", "32",
                    "--batch-size", "32", "--seed", "20260817", "--precision", "bf16",
                    "--compile", "--secondary-structure", "all_x",
                ],
                cwd=str(remote_cwd), run_dir=str(sample_dir), base_config=None,
            )
            task_id = f"sample-{step}-{variant.name}"
            sample_ids[step].append(task_id)
            tasks.append(_task(
                task_id, prepared, "sample",
                [{"task_id": train_ids[variant.name], "condition": "succeeded"}],
            ))

    for step in MILESTONES:
        analysis_dir = output_root / "analysis" / f"step{step:06d}"
        prepared = prepare_run(
            name=f"hk-local-center-ss-analysis-{step}-{short}", profile=cpu_profile,
            python_args=[
                "{cwd}/scripts/analyze_sample_panel.py",
                str(output_root / "samples" / f"step{step:06d}"),
                "--output", str(output_root / "analysis" / f"step{step:06d}.json"),
            ], cwd=str(remote_cwd), run_dir=str(analysis_dir), base_config=None,
        )
        tasks.append(_task(
            f"analysis-{step}", prepared, "cpu",
            [{"task_id": item, "condition": "succeeded"} for item in sample_ids[step]],
        ))
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
