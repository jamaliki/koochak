#!/usr/bin/env python3
"""Submit a paired L128 sampler step-scale sweep through Koochak/Scruffy."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
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
from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    BASE_CONFIG,
    CPU_PROFILE,
    ESMFOLD_PROFILE,
    GPU_PROFILE,
    KOOCHAK_COMMIT,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_COMMIT,
    SCRUFFY_ROOT,
    SCRUFFY_SITE,
    SAMPLES_PER_LENGTH,
    _assert_config,
    _git,
    _patches,
)


SOURCE_COMMIT = "4fae11513a5012b908da749104a675ed557d9590"
SOURCE_ROOT = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k" / SOURCE_COMMIT
SOURCE_STEP = 150_000
SAMPLE_SEED = 20260901
STEP_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0)
VARIANTS = (
    "pool_before_attention_pair_transition",
    "flat_after_node_no_transition",
)
VARIANT_MODEL_SETTINGS = {
    "pool_before_attention_pair_transition": ("masked_pool", "before_attention", True),
    "flat_after_node_no_transition": ("flat_linear", "after_node", False),
}
RESOURCES = {
    "sample": {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 14, "memory_gb_per_node": 128, "time_limit_seconds": 21_600},
    "esmfold": {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 8, "memory_gb_per_node": 128, "time_limit_seconds": 43_200},
    "analysis": {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 8, "memory_gb_per_node": 32, "time_limit_seconds": 14_400},
}


def _scale_tag(scale: float) -> str:
    return f"scale{scale:.2f}".replace(".", "p")


def _output(artifact_id: str, path: Path, *, stage: str, workflow: str, task: str, kind: str, expected_records: int | None = None) -> DeclaredOutput:
    return DeclaredOutput(
        artifact_id,
        str(path),
        kind=kind,
        stage=stage,
        expected_records=expected_records,
        provenance={"project_id": PROJECT_ID, "workflow_id": workflow, "task_id": task, "code_commit": _git("rev-parse", "HEAD")},
        metadata={"stage": stage, "workflow": workflow},
    )


def _stage_run(*, stage: str, task: str, workflow: str, artifact: DeclaredOutput, run_dir: Path, profile, command: list[str], base_config: Path | None = None, patches: list[ConfigPatch] | None = None):
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
        name=f"hk-step-scale-{stage}-{task}",
        profile=profile,
        python_args=arguments,
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(run_dir),
        base_config=base_config,
        patches=patches or [],
        declared_outputs=[artifact],
    )
    if base_config is not None:
        _assert_config(run, patches or [])
    return run


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-step-scale-150k-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-step-scale-150k" / code_commit
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    profiles = {
        "gpu": load_environment_profile(GPU_PROFILE),
        "cpu": load_environment_profile(CPU_PROFILE),
        "esmfold": load_environment_profile(ESMFOLD_PROFILE),
    }
    tasks: list[PreparedTask] = []
    sample_ids: dict[tuple[str, float], str] = {}
    for scale in STEP_SCALES:
        tag = _scale_tag(scale)
        for variant in VARIANTS:
            train_dir = SOURCE_ROOT / "train" / "L128" / variant
            checkpoint = train_dir / f"step{SOURCE_STEP:09d}.pt"
            sample_id = f"sample-{tag}-{variant}"
            sample_ids[(variant, scale)] = sample_id
            sample_dir = output_root / "samples" / tag / variant
            sample_output = _output(
                f"samples/{tag}/{variant}", sample_dir, stage="sample", workflow=workflow,
                task=sample_id, kind="directory", expected_records=SAMPLES_PER_LENGTH,
            )
            patches = [
                *_patches(
                    variant=(variant, *VARIANT_MODEL_SETTINGS[variant]),
                    length=128,
                    run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
                    workflow=workflow,
                    max_steps=SOURCE_STEP,
                ),
                ConfigPatch("sampling.step_scale", scale),
            ]
            sample = _stage_run(
                stage="sample", task=sample_id, workflow=workflow, artifact=sample_output,
                run_dir=sample_dir.with_name(sample_dir.name + ".managed"), profile=profiles["gpu"],
                base_config=BASE_CONFIG, patches=patches,
                command=[
                    "{cwd}/scripts/sample_short128_milestone.py", "--config", "{config}",
                    "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
                    "--lengths", "128", "--samples-per-length", str(SAMPLES_PER_LENGTH),
                    "--batch-size", "8", "--seed", str(SAMPLE_SEED), "--precision", "bf16", "--compile",
                ],
            )
            tasks.append(PreparedTask(sample_id, sample, RESOURCES["sample"], recovery=RECOVERY))

        esmfold_root = output_root / "esmfold" / tag
        esmfold_waits = []
        for variant in VARIANTS:
            sample_id = sample_ids[(variant, scale)]
            sample_dir = output_root / "samples" / tag / variant
            esmfold_id = f"esmfold-{tag}-{variant}"
            esmfold_dir = esmfold_root / variant / "L0128"
            esmfold_output = _output(
                f"esmfold/{tag}/{variant}/L0128", esmfold_dir, stage="esmfold", workflow=workflow,
                task=esmfold_id, kind="directory", expected_records=SAMPLES_PER_LENGTH,
            )
            esmfold = _stage_run(
                stage="esmfold", task=esmfold_id, workflow=workflow, artifact=esmfold_output,
                run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"), profile=profiles["esmfold"],
                command=[
                    "{cwd}/scripts/run_esmfold_designability_shard.py",
                    "--sample-dir", str(sample_dir / "L0128"), "--output-dir", str(esmfold_dir),
                    "--variant", variant, "--step", str(SOURCE_STEP), "--length", "128",
                    "--expected-count", str(SAMPLES_PER_LENGTH), "--wrapper", "{cwd}/scripts/esmfold_predict_container",
                    "--chunk-size", "8", "--bf16",
                ],
            )
            tasks.append(PreparedTask(esmfold_id, esmfold, RESOURCES["esmfold"], wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": f"samples/{tag}/{variant}"},), recovery=RECOVERY))
            esmfold_waits.append({"kind": "artifact", "task_id": esmfold_id, "artifact_id": f"esmfold/{tag}/{variant}/L0128"})

        analysis_id = f"analysis-{tag}"
        analysis_path = output_root / "analysis" / tag / "L128" / "step_scale.json"
        analysis_output = _output(
            f"analysis/{tag}/L128/step_scale.json", analysis_path, stage="analysis", workflow=workflow,
            task=analysis_id, kind="file", expected_records=1,
        )
        analysis = _stage_run(
            stage="analysis", task=analysis_id, workflow=workflow, artifact=analysis_output,
            run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"), profile=profiles["cpu"],
            command=[
                "{cwd}/scripts/analyze_step_scale_designability.py", "--esmfold-root", str(esmfold_root),
                "--variants", ",".join(VARIANTS), "--step", str(SOURCE_STEP), "--step-scale", str(scale),
                "--expected-count", str(SAMPLES_PER_LENGTH), "--output", str(analysis_path),
            ],
        )
        tasks.append(PreparedTask(analysis_id, analysis, RESOURCES["analysis"], wait_for=tuple(esmfold_waits), recovery=RECOVERY))

    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    koochak_root = (REPO_ROOT / "external/koochak").resolve()
    if _git("rev-parse", "HEAD", cwd=koochak_root) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from the independent checkout {expected}")
    checkpoints = [SOURCE_ROOT / "train" / "L128" / variant / f"step{SOURCE_STEP:09d}.pt" for variant in VARIANTS]
    missing = [str(item) for item in checkpoints if not item.is_file()]
    if missing:
        raise RuntimeError(f"source checkpoints are missing: {missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _validate_online(code_commit)
    workflow = build_workflow(code_commit)
    description = {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "project_id": workflow.project_id,
        "code_commit": code_commit,
        "source_commit": SOURCE_COMMIT,
        "source_step": SOURCE_STEP,
        "step_scales": list(STEP_SCALES),
        "variants": list(VARIANTS),
        "task_count": len(workflow.tasks),
        "task_counts": Counter("gpu" if task.resources["gpus_per_node"] else "cpu" for task in workflow.tasks),
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    result = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "submission": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
