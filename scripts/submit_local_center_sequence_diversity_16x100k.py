#!/usr/bin/env python3
"""Materialize and submit the local-center sequence-diversity campaign."""

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
    stage_run,
    submit_scruffy,
)
import koochak  # noqa: E402


BASE_MAIN_COMMIT = "908ae77"
KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site")
PROJECT_ID = "kaveh-ce20-20260806"
BASE_CONFIG = REPO_ROOT / "configs/experiments/local_center_sequence_diversity_16x100k.yaml"
GPU_PROFILE = REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml"
CPU_PROFILE = REPO_ROOT / "environments/tokyo-pair-distogram-cpu.yaml"
MILESTONES = (10_000, 25_000, 50_000, 100_000)


@dataclass(frozen=True)
class SequenceVariant:
    name: str
    sigma_max: float
    ramp_max: float | None
    polar_weight: float
    marginal_js_weight: float


VARIANTS = (
    *(SequenceVariant(f"hard05_{polar}_{js}", 0.5, None, weight, js_weight)
      for polar, weight in (("uniform", 1.0), ("polar2", 2.0))
      for js, js_weight in (("nojs", 0.0), ("js005", 0.05))),
    *(SequenceVariant(f"hard10_{polar}_{js}", 1.0, None, weight, js_weight)
      for polar, weight in (("uniform", 1.0), ("polar2", 2.0))
      for js, js_weight in (("nojs", 0.0), ("js005", 0.05))),
    *(SequenceVariant(f"lin05to10_{polar}_{js}", 0.5, 1.0, weight, js_weight)
      for polar, weight in (("uniform", 1.0), ("polar2", 2.0))
      for js, js_weight in (("nojs", 0.0), ("js005", 0.05))),
    *(SequenceVariant(f"lin05to20_{polar}_{js}", 0.5, 2.0, weight, js_weight)
      for polar, weight in (("uniform", 1.0), ("polar2", 2.0))
      for js, js_weight in (("nojs", 0.0), ("js005", 0.05))),
)
assert tuple(item.name for item in VARIANTS) == (
    "hard05_uniform_nojs", "hard05_uniform_js005", "hard05_polar2_nojs", "hard05_polar2_js005",
    "hard10_uniform_nojs", "hard10_uniform_js005", "hard10_polar2_nojs", "hard10_polar2_js005",
    "lin05to10_uniform_nojs", "lin05to10_uniform_js005", "lin05to10_polar2_nojs", "lin05to10_polar2_js005",
    "lin05to20_uniform_nojs", "lin05to20_uniform_js005", "lin05to20_polar2_nojs", "lin05to20_polar2_js005",
)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
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
    if not SCRUFFY_ROOT.exists() or not SCRUFFY_SITE.exists():
        raise RuntimeError("Scruffy launch paths are unavailable")
    return commit


def _patches(variant: SequenceVariant, run_dir: Path, *, production: bool) -> list[ConfigPatch]:
    patches = [
        ConfigPatch("loss.aatype_sigma_max", variant.sigma_max),
        ConfigPatch("loss.aatype_sigma_ramp_max", variant.ramp_max),
        ConfigPatch("loss.polar_weight", variant.polar_weight),
        ConfigPatch("loss.aatype_marginal_js_weight", variant.marginal_js_weight),
        ConfigPatch("logging.csv_path", str(run_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "log.jsonl")),
        ConfigPatch("wandb.name", f"local-center-seq-16x100k-{variant.name}"),
        ConfigPatch("wandb.tags", ["local-center-seq-16x100k", variant.name]),
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
        "loss.aatype_sigma_max", "loss.aatype_sigma_ramp_max", "loss.polar_weight",
        "loss.aatype_marginal_js_weight", "logging.csv_path", "logging.jsonl_path",
        "train.out_dir", "wandb.name", "wandb.tags", "wandb.enabled", "wandb.mode",
        "wandb.resume",
    }
    if any(patch.path not in allowed for patch in patches):
        raise AssertionError("launcher contains a patch outside the campaign contract")
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    actual = OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)
    expected = OmegaConf.load(BASE_CONFIG)
    OmegaConf.update(expected, "train.out_dir", prepared.run_dir, force_add=True)
    for patch in patches:
        OmegaConf.update(expected, patch.path, patch.value, merge=patch.merge, force_add=True)
    expected = OmegaConf.to_container(expected, resolve=True)
    if actual != expected:
        raise AssertionError(f"rendered config differs from its allowed patches: {prepared.name}")


def _task(task_id: str, prepared, resource: str, needs: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {"task_id": task_id, "run": prepared, "resource": resource, "needs": needs or []}


def _prepare_tasks(commit: str) -> tuple[str, list[dict[str, object]]]:
    short = commit[:7]
    output_root = REMOTE_RUN_ROOT / "local-center-seq-16x100k" / commit
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    gpu_profile = load_environment_profile(GPU_PROFILE)
    cpu_profile = load_environment_profile(CPU_PROFILE)
    workflow_id = f"hk-local-center-seq-16x100k-{short}-v1"
    tasks: list[dict[str, object]] = []
    preflight_ids = []
    for variant in VARIANTS:
        run_dir = output_root / "preflight" / variant.name
        prepared = prepare_run(
            name=f"hk-seq-preflight-{variant.name}-{short}", profile=gpu_profile,
            python_args=["{cwd}/scripts/profile_short128_latency.py", "--base-config", "{config}",
                         "--run-dir", "{run_dir}/profile", "--steps", "2", "--warmup", "1",
                         "--cache-size", "1", "--summary-output", "{run_dir}/latency.json"],
            cwd=str(remote_cwd), run_dir=str(run_dir), base_config=BASE_CONFIG,
            patches=_patches(variant, run_dir, production=False),
        )
        _assert_rendered_config(prepared, _patches(variant, run_dir, production=False))
        task_id = f"preflight-{variant.name}"
        preflight_ids.append(task_id)
        tasks.append(_task(task_id, prepared, "preflight"))

    canary_ids = []
    for variant in (VARIANTS[0], VARIANTS[-1]):
        run_dir = output_root / "canary" / variant.name
        prepared = prepare_run(
            name=f"hk-seq-canary-{variant.name}-{short}", profile=gpu_profile,
            python_args=["{cwd}/scripts/profile_short128_latency.py", "--base-config", "{config}",
                         "--run-dir", "{run_dir}/profile", "--steps", "250", "--warmup", "100",
                         "--summary-output", "{run_dir}/latency.json"],
            cwd=str(remote_cwd), run_dir=str(run_dir), base_config=BASE_CONFIG,
            patches=_patches(variant, run_dir, production=False),
        )
        _assert_rendered_config(prepared, _patches(variant, run_dir, production=False))
        task_id = f"canary-{variant.name}"
        canary_ids.append(task_id)
        tasks.append(_task(task_id, prepared, "canary", [{"task_id": item, "condition": "succeeded"} for item in preflight_ids]))

    guard_dir = output_root / "throughput-guard"
    guard_args = ["{cwd}/scripts/check_throughput_guard.py", "--baseline",
                  str(output_root / "canary" / VARIANTS[0].name / "latency.json"), "--output",
                  str(guard_dir / "report.json"), "--minimum-rows", "100", "--max-p50-regression", "0.08",
                  "--max-p95-regression", "0.12"]
    guard_args.extend(["--candidate", str(output_root / "canary" / VARIANTS[-1].name / "latency.json")])
    # The guard is short-lived but still needs a reconciled Slurm step.  The
    # GPU profile is the validated Koochak launch path for this Scruffy queue;
    # keep the explicit one-GPU guard reservation below rather than allowing a
    # CPU-profile process to exit before Scruffy can reconcile placement.
    guard = prepare_run(name=f"hk-seq-throughput-guard-{short}", profile=gpu_profile,
                        python_args=guard_args, cwd=str(remote_cwd), run_dir=str(guard_dir), base_config=None)
    tasks.append(_task("throughput-guard", guard, "cpu", [{"task_id": item, "condition": "succeeded"} for item in canary_ids]))

    train_ids = []
    train_dirs = {}
    train_needs = [{"task_id": "throughput-guard", "condition": "succeeded"}]
    train_needs.extend({"task_id": item, "condition": "succeeded"} for item in preflight_ids)
    for variant in VARIANTS:
        run_dir = output_root / "train" / variant.name
        train_dirs[variant.name] = run_dir
        prepared = prepare_run(
            name=f"hk-seq-train-{variant.name}-{short}", profile=gpu_profile,
            python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}"],
            cwd=str(remote_cwd), run_dir=str(run_dir), base_config=BASE_CONFIG,
            patches=_patches(variant, run_dir, production=True),
        )
        _assert_rendered_config(prepared, _patches(variant, run_dir, production=True))
        task_id = f"train-{variant.name}"
        train_ids.append(task_id)
        tasks.append(_task(task_id, prepared, "train", train_needs))

    sample_ids_by_step = {step: [] for step in MILESTONES}
    for step in MILESTONES:
        for variant in VARIANTS:
            run_dir = output_root / "samples" / f"step{step:06d}" / variant.name
            train_dir = train_dirs[variant.name]
            width = 7 if step == 100_000 else 9
            checkpoint = train_dir / f"step{step:0{width}d}.pt"
            prepared = prepare_run(
                name=f"hk-seq-sample-{step}-{variant.name}-{short}", profile=gpu_profile,
                python_args=["{cwd}/scripts/sample_short128_milestone.py", "--config", str(train_dir / "config.yaml"),
                             "--checkpoint", str(checkpoint), "--output-dir", str(run_dir), "--lengths", "64,96,128",
                             "--samples-per-length", "32", "--batch-size", "32", "--seed", "20260813",
                             "--precision", "bf16", "--compile"],
                cwd=str(remote_cwd), run_dir=str(run_dir), base_config=None,
            )
            task_id = f"sample-{step}-{variant.name}"
            sample_ids_by_step[step].append(task_id)
            tasks.append(_task(task_id, prepared, "sample", [{"task_id": f"train-{variant.name}", "condition": "succeeded"}]))

    analysis_ids = []
    for step in MILESTONES:
        run_dir = output_root / "analysis" / f"step{step:06d}"
        output = output_root / "analysis" / f"milestone_step{step:06d}.json"
        prepared = prepare_run(
            name=f"hk-seq-analysis-{step}-{short}", profile=cpu_profile,
            python_args=["{cwd}/scripts/analyze_sequence_diversity_16x100k.py", "--sample-root",
                         str(output_root / "samples" / f"step{step:06d}"), "--step", str(step), "--output", str(output)],
            cwd=str(remote_cwd), run_dir=str(run_dir), base_config=None,
        )
        task_id = f"analysis-{step}"
        analysis_ids.append(task_id)
        tasks.append(_task(task_id, prepared, "cpu", [{"task_id": item, "condition": "succeeded"} for item in sample_ids_by_step[step]]))

    final_dir = output_root / "analysis" / "final"
    final = prepare_run(
        name=f"hk-seq-analysis-final-{short}", profile=cpu_profile,
        python_args=["{cwd}/scripts/analyze_sequence_diversity_16x100k.py", "--finalize",
                     "--analysis-dir", str(output_root / "analysis"), "--output",
                     str(output_root / "analysis" / "final_comparison.md")],
        cwd=str(remote_cwd), run_dir=str(final_dir), base_config=None,
    )
    tasks.append(_task("analysis-final", final, "cpu", [{"task_id": item, "condition": "succeeded"} for item in analysis_ids]))
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
        result = [{"task_id": item["task_id"], "name": item["run"].name,
                   "run_dir": item["run"].run_dir, "resource": item["resource"], "needs": item["needs"]} for item in tasks]
    elif args.stage_only:
        for item in tasks:
            stage_run(item["run"])
        result = [{"task_id": item["task_id"], "staged": True} for item in tasks]
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest  # noqa: PLC0415

        resource_values = {
            "preflight": (1, 14, 240, 7_200), "canary": (1, 14, 240, 14_400),
            "train": (1, 14, 240, 172_800), "sample": (1, 14, 128, 43_200),
            # Scruffy requires a reconciled Slurm step for runtime placement;
            # reserve one GPU even though these short checks are CPU-bound.
            "cpu": (1, 2, 16, 3_600),
        }
        result = []
        for item in tasks:
            gpus, cpus, memory, seconds = resource_values[item["resource"]]
            submitted = submit_scruffy(
                item["run"], root=SCRUFFY_ROOT,
                resources=ResourceRequest(nodes=1, gpus_per_node=gpus, cpus_per_node=cpus,
                                          memory_gb_per_node=memory, time_limit_seconds=seconds),
                request_id=f"{workflow_id}/{item['task_id']}/v1", project_id=PROJECT_ID,
                workflow_id=workflow_id, task_id=item["task_id"], needs=item["needs"],
            )
            result.append({"task_id": item["task_id"], **submitted})
    print(json.dumps({"workflow_id": workflow_id, "task_count": len(tasks), "tasks": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
