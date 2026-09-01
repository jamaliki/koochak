#!/usr/bin/env python3
"""Submit the 8-cell patch/coarse factorial for L128 and L256 through Scruffy."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    DeclaredOutput,
    PreparedTask,
    PreparedWorkflow,
    load_environment_profile,
    prepare_run,
    submit_scruffy_workflow,
)


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
KOOCHAK_COMMIT = "eec841f0261c77c31d55100b1dd8af1e753903f5"
SCRUFFY_COMMIT = "d60afabf82693c8c8e2108439696526c6bd2bd32"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path(
    "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site"
)
BASE_CONFIG = REPO_ROOT / "configs/train.yaml"
METADATA = Path(
    "/mnt/lustre/users/kiarash-eitgbi/atom14/afdb_all_parsed/"
    "parsed_np_shards_with_ss_3di/metadata_ca4_patch4.json"
)
GPU_PROFILE = REPO_ROOT / "environments/tokyo-factorial-gpu.yaml"
CPU_PROFILE = REPO_ROOT / "environments/tokyo-factorial-cpu.yaml"
ESMFOLD_PROFILE = REPO_ROOT / "environments/tokyo-factorial-esmfold.yaml"
MILESTONES = tuple(range(50_000, 500_001, 50_000))
SAMPLES_PER_LENGTH = 32
SAMPLE_SEED = 20260901
VARIANTS = (
    ("pool_after_node_no_transition", "masked_pool", "after_node", False),
    ("pool_after_node_pair_transition", "masked_pool", "after_node", True),
    ("pool_before_attention_no_transition", "masked_pool", "before_attention", False),
    ("pool_before_attention_pair_transition", "masked_pool", "before_attention", True),
    ("flat_after_node_no_transition", "flat_linear", "after_node", False),
    ("flat_after_node_pair_transition", "flat_linear", "after_node", True),
    ("flat_before_attention_no_transition", "flat_linear", "before_attention", False),
    ("flat_before_attention_pair_transition", "flat_linear", "before_attention", True),
)
LENGTHS = (128, 256)

RESOURCES = {
    "train": {"nodes": 1, "gpus_per_node": 8, "cpus_per_node": 112, "memory_gb_per_node": 512, "time_limit_seconds": 259_200},
    "sample": {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 14, "memory_gb_per_node": 128, "time_limit_seconds": 21_600},
    "esmfold": {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 8, "memory_gb_per_node": 128, "time_limit_seconds": 43_200},
    "analysis": {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 8, "memory_gb_per_node": 32, "time_limit_seconds": 14_400},
}
RECOVERY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
    "evacuation": {"signal": "USR1", "grace_seconds": 600},
}


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _tag(step: int) -> str:
    return f"step{step:09d}"


def _patches(
    *, variant: tuple[str, str, str, bool], length: int, run_dir: Path, workflow: str,
    batch_size: int, grad_accum: int, max_steps: int = 500_000,
) -> list[ConfigPatch]:
    name, patchify, pair_position, pair_transition = variant
    patch_capacity = (length + 3) // 4
    return [
        ConfigPatch("model.patchify_mode", patchify),
        ConfigPatch("model.coarse_pair_position", pair_position),
        ConfigPatch("model.coarse_pair_transition", pair_transition),
        ConfigPatch("data.metadata_path", str(METADATA)),
        ConfigPatch("data.max_length", length),
        ConfigPatch("data.length_buckets", [length]),
        ConfigPatch("data.patch_capacities", [patch_capacity]),
        ConfigPatch("data.batch_size", batch_size),
        ConfigPatch("train.grad_accum", grad_accum),
        ConfigPatch("train.max_steps", max_steps),
        ConfigPatch("train.ckpt_every", 50_000),
        ConfigPatch("train.keep_last_k", 12),
        ConfigPatch("train.evacuation_enabled", True),
        ConfigPatch("logging.csv_path", str(run_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "log.jsonl")),
        ConfigPatch("wandb.enabled", False),
        ConfigPatch("wandb.mode", "disabled"),
    ]


def _disabled_wandb() -> list[ConfigPatch]:
    return [ConfigPatch("wandb.enabled", False), ConfigPatch("wandb.mode", "disabled")]


def _assert_config(prepared, patches: list[ConfigPatch]) -> None:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    actual = OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)
    expected = OmegaConf.load(BASE_CONFIG)
    OmegaConf.update(expected, "train.out_dir", prepared.run_dir, force_add=True)
    for patch in patches:
        OmegaConf.update(expected, patch.path, patch.value, merge=patch.merge, force_add=True)
    if actual != OmegaConf.to_container(expected, resolve=True):
        raise AssertionError(f"rendered config differs from declared patches: {prepared.name}")


def _output(
    artifact_id: str, path: Path, *, stage: str, workflow: str, task: str,
    kind: str, expected_records: int | None = None,
) -> DeclaredOutput:
    return DeclaredOutput(
        artifact_id, str(path), kind=kind, stage=stage, expected_records=expected_records,
        provenance={
            "project_id": PROJECT_ID, "workflow_id": workflow,
            "task_id": task, "code_commit": _git("rev-parse", "HEAD"),
        },
        metadata={"stage": stage, "workflow": workflow},
    )


def _stage_run(
    *, stage: str, task: str, workflow: str, artifact: DeclaredOutput,
    run_dir: Path, profile, command: list[str], base_config: Path | None = None,
    patches: list[ConfigPatch] | None = None,
):
    arguments = [
        "{cwd}/scripts/robust_factorial_stage.py", "--stage", stage,
        "--artifact-id", artifact.artifact_id, "--artifact-path", artifact.path,
        "--kind", artifact.kind, "--project", PROJECT_ID, "--workflow", workflow,
        "--task", task, "--code-commit", _git("rev-parse", "HEAD"),
    ]
    if artifact.expected_records is not None:
        arguments.extend(["--expected-records", str(artifact.expected_records)])
    arguments.extend(["--", *command])
    run = prepare_run(
        name=f"hk-factorial-{stage}-{task}", profile=profile, python_args=arguments,
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(run_dir), base_config=base_config, patches=patches or [],
        declared_outputs=[artifact],
    )
    if base_config is not None:
        _assert_config(run, patches or [])
    return run


def build_workflow(code_commit: str, lengths: tuple[int, ...]) -> PreparedWorkflow:
    short = code_commit[:7]
    length_suffix = "-" + "-".join(f"L{length}" for length in lengths)
    workflow = f"hk-patch-coarse-factorial-500k{length_suffix}-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k" / code_commit
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    profiles = {
        "gpu": load_environment_profile(GPU_PROFILE),
        "cpu": load_environment_profile(CPU_PROFILE),
        "esmfold": load_environment_profile(ESMFOLD_PROFILE),
    }
    tasks: list[PreparedTask] = []
    train_ids: dict[tuple[str, int], str] = {}
    sample_ids: dict[tuple[str, int, int], str] = {}

    for variant in VARIANTS:
        for length in lengths:
            name = variant[0]
            train_id = f"train-{name}-L{length}"
            train_ids[(name, length)] = train_id
            train_dir = output_root / "train" / f"L{length}" / name
            batch_size, grad_accum = (32, 1) if length == 128 else (8, 4)
            patches = _patches(
                variant=variant, length=length, run_dir=train_dir,
                workflow=workflow, batch_size=batch_size, grad_accum=grad_accum,
            )
            train = prepare_run(
                name=f"hk-factorial-train-{name}-L{length}-{short}",
                profile=profiles["gpu"],
                python_args=[
                    "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                    "-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto",
                ],
                cwd=str(remote_cwd), run_dir=str(train_dir), base_config=BASE_CONFIG,
                patches=patches,
            )
            _assert_config(train, patches)
            tasks.append(PreparedTask(train_id, train, RESOURCES["train"], recovery=RECOVERY))

            for step in MILESTONES:
                tag = _tag(step)
                checkpoint = train_dir / f"{tag}.pt"
                checkpoint_id = f"checkpoint/{tag}.pt"
                sample_id = f"sample-{tag}-{name}-L{length}"
                sample_ids[(name, length, step)] = sample_id
                sample_dir = output_root / "samples" / tag / f"L{length}" / name
                sample_output = _output(
                    f"samples/{tag}/L{length}/{name}", sample_dir,
                    stage="sample", workflow=workflow, task=sample_id,
                    kind="directory", expected_records=SAMPLES_PER_LENGTH,
                )
                sample_patches = [*patches, *_disabled_wandb()]
                sample = _stage_run(
                    stage="sample", task=sample_id, workflow=workflow,
                    artifact=sample_output, run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
                    profile=profiles["gpu"], base_config=BASE_CONFIG,
                    patches=sample_patches,
                    command=[
                        "{cwd}/scripts/sample_short128_milestone.py", "--config", "{config}",
                        "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
                        "--lengths", str(length), "--samples-per-length", str(SAMPLES_PER_LENGTH),
                        "--batch-size", "8", "--seed", str(SAMPLE_SEED), "--precision", "bf16", "--compile",
                    ],
                )
                tasks.append(PreparedTask(
                    sample_id, sample, RESOURCES["sample"],
                    wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": checkpoint_id},),
                    recovery=RECOVERY,
                ))

                esmfold_id = f"esmfold-{tag}-{name}-L{length}"
                esmfold_dir = output_root / "esmfold" / tag / name / f"L{length:04d}"
                esmfold_output = _output(
                    f"esmfold/{tag}/{name}/L{length:04d}", esmfold_dir,
                    stage="esmfold", workflow=workflow, task=esmfold_id,
                    kind="directory", expected_records=SAMPLES_PER_LENGTH,
                )
                esmfold = _stage_run(
                    stage="esmfold", task=esmfold_id, workflow=workflow,
                    artifact=esmfold_output, run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"),
                    profile=profiles["esmfold"],
                    command=[
                        "{cwd}/scripts/run_esmfold_designability_shard.py",
                        "--sample-dir", str(sample_dir / f"L{length:04d}"),
                        "--output-dir", str(esmfold_dir),
                        "--variant", name, "--step", str(step), "--length", str(length),
                        "--expected-count", str(SAMPLES_PER_LENGTH),
                        "--wrapper", "{cwd}/scripts/esmfold_predict_container",
                        "--chunk-size", "64", "--bf16",
                    ],
                )
                tasks.append(PreparedTask(
                    esmfold_id, esmfold, RESOURCES["esmfold"],
                    wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},),
                    recovery=RECOVERY,
                ))

    variant_names = ",".join(item[0] for item in VARIANTS)
    for step in MILESTONES:
        tag = _tag(step)
        for length in lengths:
            analysis_id = f"analysis-{tag}-L{length}"
            analysis_path = output_root / "analysis" / tag / f"L{length}" / "milestone.json"
            analysis_output = _output(
                f"analysis/{tag}/L{length}/milestone.json", analysis_path,
                stage="analysis", workflow=workflow, task=analysis_id,
                kind="file", expected_records=1,
            )
            analysis = _stage_run(
                stage="analysis", task=analysis_id, workflow=workflow,
                artifact=analysis_output,
                run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"),
                profile=profiles["cpu"],
                command=[
                    "{cwd}/scripts/analyze_esmfold_designability.py", "milestone",
                    "--input-root", str(output_root / "esmfold" / tag), "--step", str(step),
                    "--variants", variant_names, "--lengths", str(length),
                    "--expected-per-length", str(SAMPLES_PER_LENGTH),
                    "--reference", VARIANTS[0][0], "--output", str(analysis_path),
                ],
            )
            waits = tuple(
                {
                    "kind": "artifact", "task_id": f"esmfold-{tag}-{variant[0]}-L{length}",
                    "artifact_id": f"esmfold/{tag}/{variant[0]}/L{length:04d}",
                }
                for variant in VARIANTS
            )
            tasks.append(PreparedTask(analysis_id, analysis, RESOURCES["analysis"], wait_for=waits, recovery=RECOVERY))

    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )


def _describe(
    workflow: PreparedWorkflow, code_commit: str, lengths: tuple[int, ...]
) -> dict[str, object]:
    return {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "project_id": workflow.project_id,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "milestones": list(MILESTONES),
        "lengths": list(lengths),
        "variant_count": len(VARIANTS),
        "task_count": len(workflow.tasks),
        "task_counts": Counter(
            "gpu" if task.resources["gpus_per_node"] else "cpu" for task in workflow.tasks
        ),
        "tasks": [
            {
                "task_id": task.task_id, "resources": dict(task.resources),
                "needs": list(task.needs), "wait_for": list(task.wait_for),
                "outputs": [output.artifact_id for output in task.run.declared_outputs],
            }
            for task in workflow.tasks
        ],
    }


def _validate_online(code_commit: str) -> None:
    koochak_root = (REPO_ROOT / "external/koochak").resolve()
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    if _git("rev-parse", "HEAD", cwd=koochak_root) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from the independent checkout {expected}")
    missing = [str(item) for item in (SCRUFFY_ROOT, SCRUFFY_SITE, METADATA) if not item.exists()]
    if missing:
        raise RuntimeError(f"required launch locations are missing: {missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _validate_online(code_commit)
    workflows = tuple(build_workflow(code_commit, (length,)) for length in LENGTHS)
    descriptions = [
        _describe(workflow, code_commit, (length,))
        for workflow, length in zip(workflows, LENGTHS)
    ]
    if args.dry_run:
        print(json.dumps({"workflows": descriptions, "total_task_count": sum(item["task_count"] for item in descriptions)}, indent=2, sort_keys=True, default=str))
        return
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    submissions = []
    for workflow, description in zip(workflows, descriptions):
        submissions.append({
            "workflow": description,
            "submission": submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT),
        })
    print(json.dumps({"workflows": submissions}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
