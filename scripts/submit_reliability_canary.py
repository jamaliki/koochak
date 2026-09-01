#!/usr/bin/env python3
"""Build and optionally submit the one-arm Koochak reliability canary.

The dry-run is a pure serialization of the prepared workflow.  It deliberately
does not stage launch manifests or import a Scruffy client.  The online path
has one admission call for the complete train -> sample -> fold -> analysis
DAG; all downstream edges are strict artifact conditions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    DeclaredOutput,
    EnvironmentProfile,
    PreparedTask,
    PreparedWorkflow,
    prepare_run,
    submit_scruffy_workflow,
)


PROJECT_ID = "hierarchical-kaveh-reliability-canary"
KOOCHAK_COMMIT = "2c64510098c78a98984fff133d4a2a6de0eda8c4"
SCRUFFY_COMMIT = "d2b7dc2f98794eaf585077f67b9fd3644bb565ab"
BASE_CONFIG = REPO_ROOT / "configs" / "train.yaml"
DEFAULT_METADATA = Path(
    "/mnt/lustre/users/kiarash-eitgbi/atom14/afdb_all_parsed/"
    "parsed_np_shards_with_ss_3di/metadata_ca4_patch4.json"
)
CANARY_TRAIN_LENGTH = 64
MILESTONE_STEPS = (2, 4)
CHECKPOINT_ARTIFACTS = tuple(
    f"checkpoint/step{step:09d}.pt" for step in MILESTONE_STEPS
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _profile(
    *, python: str, gpu: bool, artifact_ack_timeout_s: int | None = None
) -> EnvironmentProfile:
    path_entries = [str(Path(python).parent), "/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    variables = {
        "PATH": ":".join(path_entries),
        "PYTHONPATH": "{cwd}/external/koochak:{cwd}",
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "online" if gpu else "disabled",
    }
    if not gpu:
        variables["WANDB_DISABLED"] = "true"
    if artifact_ack_timeout_s is not None:
        variables["KOOCHAK_SCRUFFY_ARTIFACT_ACK_TIMEOUT_SECONDS"] = str(artifact_ack_timeout_s)
    return EnvironmentProfile(
        profile_id="hierarchical-kaveh-reliability-canary",
        python=python,
        variables=variables,
        packages={"koochak": "*", "numpy": "*", "omegaconf": "*"},
    )


def _patches(
    *,
    metadata: Path,
    train_dir: Path,
    wandb_enabled: bool,
    workflow_id: str,
) -> list[ConfigPatch]:
    patches: list[ConfigPatch] = [
        ConfigPatch("model.node_dim", 32),
        ConfigPatch("model.condition_dim", 16),
        ConfigPatch("model.pair_dim", 8),
        ConfigPatch("model.atom_dim", 16),
        ConfigPatch("model.attention_heads", 4),
        ConfigPatch("model.attention_head_dim", 8),
        ConfigPatch("model.atom_heads", 2),
        ConfigPatch("model.atom_head_dim", 8),
        ConfigPatch("model.atom_encoder_depth", 1),
        ConfigPatch("model.residue_encoder_depth", 1),
        ConfigPatch("model.coarse_depth", 1),
        ConfigPatch("model.residue_decoder_depth", 1),
        ConfigPatch("model.atom_decoder_depth", 1),
        ConfigPatch("model.residue_ffn_expansion", 2),
        ConfigPatch("model.atom_ffn_expansion", 2),
        ConfigPatch("model.pair_rbf_bins", 4),
        ConfigPatch("model.distogram_bins", 8),
        ConfigPatch("data.metadata_path", str(metadata)),
        ConfigPatch("data.min_length", 4),
        ConfigPatch("data.max_length", CANARY_TRAIN_LENGTH),
        ConfigPatch("data.length_buckets", [CANARY_TRAIN_LENGTH]),
        ConfigPatch("data.batch_size", 1),
        ConfigPatch("data.num_workers", 0),
        ConfigPatch("data.pin_memory", False),
        ConfigPatch("data.persistent_workers", False),
        ConfigPatch("data.prefetch_factor", 1),
        ConfigPatch("train.max_steps", 4),
        ConfigPatch("train.log_every", 1),
        ConfigPatch("train.ckpt_every", 2),
        ConfigPatch("train.save_final", False),
        ConfigPatch("train.ddp", False),
        ConfigPatch("train.device", "cuda"),
        ConfigPatch("train.amp", "bf16"),
        ConfigPatch("train.prefetch_batches", 0),
        ConfigPatch("train.prefetch_threaded", False),
        ConfigPatch("train.prefetch_pipeline", "single"),
        ConfigPatch("train.compile.enabled", False),
        ConfigPatch("train.require_compile", False),
        ConfigPatch("train.require_fused", False),
        ConfigPatch("train.evacuation_enabled", True),
        ConfigPatch("train.evacuation_signal", "USR1"),
        ConfigPatch("train.ema.enabled", False),
        ConfigPatch("logging.csv_path", str(train_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(train_dir / "log.jsonl")),
        ConfigPatch("wandb.enabled", wandb_enabled),
        ConfigPatch("wandb.project", PROJECT_ID),
        ConfigPatch("wandb.name", f"{workflow_id}-trainer"),
        ConfigPatch("wandb.id", f"{workflow_id}-trainer"),
        ConfigPatch("wandb.resume", "allow"),
    ]
    return patches


def _output(
    artifact_id: str,
    path: Path,
    *,
    stage: str,
    project: str,
    workflow: str,
    task: str,
    code_commit: str,
    expected_records: int,
    kind: str,
) -> DeclaredOutput:
    return DeclaredOutput(
        artifact_id,
        str(path),
        kind=kind,
        stage=stage,
        provenance={
            "project_id": project,
            "workflow_id": workflow,
            "task_id": task,
            "code_commit": code_commit,
        },
        expected_records=expected_records,
        metadata={"canary": True, "stage": stage},
    )


def build_workflow(
    *,
    code_commit: str,
    run_root: Path,
    metadata: Path,
    python: str,
    wandb_enabled: bool = True,
) -> PreparedWorkflow:
    """Prepare the complete canary DAG without filesystem side effects."""

    workflow_id = f"hk-reliability-canary-{code_commit[:12]}"
    request_id = f"{PROJECT_ID}/{workflow_id}/v1"
    remote_cwd = REPO_ROOT
    train_dir = run_root / "train"
    sample_dir = run_root / "samples"
    fold_path = run_root / "fold" / "fold.json"
    analysis_path = run_root / "analysis" / "analysis.json"
    train = prepare_run(
        name=f"hk-reliability-train-{code_commit[:12]}",
        profile=_profile(python=python, gpu=True, artifact_ack_timeout_s=30),
        python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
        cwd=str(remote_cwd),
        run_dir=str(train_dir),
        base_config=BASE_CONFIG,
        patches=_patches(
            metadata=metadata,
            train_dir=train_dir,
            wandb_enabled=wandb_enabled,
            workflow_id=workflow_id,
        ),
    )
    sample_task = "sampler"
    fold_task = "fold"
    analysis_task = "analysis"
    sample_output = _output(
        "samples/canary",
        sample_dir,
        stage="sample",
        project=PROJECT_ID,
        workflow=workflow_id,
        task=sample_task,
        code_commit=code_commit,
        expected_records=1,
        kind="directory",
    )
    sample = prepare_run(
        name=f"hk-reliability-sample-{code_commit[:12]}",
        profile=_profile(python=python, gpu=True),
        python_args=[
            "{cwd}/scripts/reliability_canary_stage.py",
            "--stage", "sample",
            "--config", str(train_dir / "config.yaml"),
            "--checkpoint", str(train_dir / CHECKPOINT_ARTIFACTS[0].split("/", 1)[1]),
            "--artifact-id", sample_output.artifact_id,
            "--artifact-path", sample_output.path,
            "--kind", sample_output.kind,
            "--expected-records", "1",
            "--project", PROJECT_ID,
            "--workflow", workflow_id,
            "--task", sample_task,
            "--code-commit", code_commit,
            "--lengths", "8",
            "--samples-per-length", "1",
            "--device", "cuda",
            "--precision", "bf16",
            "--raw",
        ],
        cwd=str(remote_cwd),
        run_dir=str(run_root / "managed" / "sampler"),
        declared_outputs=[sample_output],
    )
    fold_output = _output(
        "fold/canary",
        fold_path,
        stage="fold",
        project=PROJECT_ID,
        workflow=workflow_id,
        task=fold_task,
        code_commit=code_commit,
        expected_records=1,
        kind="file",
    )
    fold = prepare_run(
        name=f"hk-reliability-fold-{code_commit[:12]}",
        profile=_profile(python=python, gpu=False),
        python_args=[
            "{cwd}/scripts/reliability_canary_stage.py", "--stage", "fold",
            "--input-path", str(sample_dir), "--artifact-id", fold_output.artifact_id,
            "--artifact-path", fold_output.path, "--kind", "file", "--expected-records", "1",
            "--project", PROJECT_ID, "--workflow", workflow_id, "--task", fold_task,
            "--code-commit", code_commit,
        ],
        cwd=str(remote_cwd),
        run_dir=str(run_root / "fold" / "run"),
        declared_outputs=[fold_output],
    )
    analysis_output = _output(
        "analysis/canary",
        analysis_path,
        stage="analysis",
        project=PROJECT_ID,
        workflow=workflow_id,
        task=analysis_task,
        code_commit=code_commit,
        expected_records=1,
        kind="file",
    )
    analysis = prepare_run(
        name=f"hk-reliability-analysis-{code_commit[:12]}",
        profile=_profile(python=python, gpu=False),
        python_args=[
            "{cwd}/scripts/reliability_canary_stage.py", "--stage", "analysis",
            "--input-path", fold_output.path, "--artifact-id", analysis_output.artifact_id,
            "--artifact-path", analysis_output.path, "--kind", "file", "--expected-records", "1",
            "--project", PROJECT_ID, "--workflow", workflow_id, "--task", analysis_task,
            "--code-commit", code_commit,
        ],
        cwd=str(remote_cwd),
        run_dir=str(run_root / "analysis" / "run"),
        declared_outputs=[analysis_output],
    )
    resources_gpu = {
        "nodes": 1, "gpus_per_node": 1, "cpus_per_node": 4,
        "memory_gb_per_node": 32, "time_limit_seconds": 3_600,
    }
    resources_cpu = {
        "nodes": 1, "gpus_per_node": 0, "cpus_per_node": 2,
        "memory_gb_per_node": 4, "time_limit_seconds": 1_800,
    }
    recovery = {
        "max_attempts": 3,
        "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
        "evacuation": {"signal": "USR1", "grace_seconds": 300},
    }
    return PreparedWorkflow(
        request_id=request_id,
        workflow_id=workflow_id,
        project_id=PROJECT_ID,
        tasks=(
            PreparedTask("trainer", train, resources_gpu, recovery=recovery),
            PreparedTask(
                sample_task,
                sample,
                resources_gpu,
                wait_for=({"kind": "artifact", "task_id": "trainer", "artifact_id": CHECKPOINT_ARTIFACTS[0]},),
            ),
            PreparedTask(
                fold_task,
                fold,
                resources_cpu,
                wait_for=({"kind": "artifact", "task_id": sample_task, "artifact_id": sample_output.artifact_id},),
            ),
            PreparedTask(
                analysis_task,
                analysis,
                resources_cpu,
                wait_for=({"kind": "artifact", "task_id": fold_task, "artifact_id": fold_output.artifact_id},),
            ),
        ),
    )


def _describe(workflow: PreparedWorkflow, *, code_commit: str) -> dict[str, Any]:
    tasks = []
    for task in workflow.tasks:
        spec = task.to_scruffy_spec(
            request_id=workflow.request_id,
            workflow_id=workflow.workflow_id,
            project_id=workflow.project_id,
        )
        tasks.append(
            {
                "task_id": task.task_id,
                "argv": spec["argv"],
                "cwd": spec["cwd"],
                "run_dir": task.run.run_dir,
                "resources": spec["resources"],
                "needs": spec["needs"],
                "wait_for": spec["wait_for"],
                "artifact_ids": [output.artifact_id for output in task.run.declared_outputs],
                "recovery": spec.get("recovery"),
                "manifest_path": task.run.manifest_path,
            }
        )
    return {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "project_id": workflow.project_id,
        "commits": {
            "hierarchical_kaveh": code_commit,
            "koochak": KOOCHAK_COMMIT,
            "scruffy": SCRUFFY_COMMIT,
        },
        "checkpoint_milestones": list(CHECKPOINT_ARTIFACTS),
        "tasks": tasks,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/lustre/users/kiarash-eitgbi/reliability-canary"))
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--python", default="/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/kaveh-koochak-8069043/bin/python")
    parser.add_argument("--scruffy-root", type=Path, default=Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105"))
    parser.add_argument("--enable-wandb", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git_commit()
    workflow = build_workflow(
        code_commit=code_commit,
        run_root=args.run_root,
        metadata=args.metadata,
        python=args.python,
        wandb_enabled=args.enable_wandb,
    )
    if args.dry_run:
        print(json.dumps(_describe(workflow, code_commit=code_commit), indent=2, sort_keys=True))
        return
    expected_koochak = (REPO_ROOT / "external" / "koochak").resolve()
    observed_koochak = subprocess.run(
        ["git", "-C", str(expected_koochak), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_koochak != KOOCHAK_COMMIT:
        raise RuntimeError(f"canary requires Koochak {KOOCHAK_COMMIT}, found {observed_koochak}")
    try:
        # Import Scruffy only after the dry-run return: local planning must not
        # depend on, inspect, or mutate a scheduler installation.
        from scruffy import status  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - fail closed at the admission boundary
        raise RuntimeError("cannot attest the Scruffy controller release") from exc
    try:
        snapshot = status(args.scruffy_root)
        if not isinstance(snapshot, Mapping):
            raise ValueError("status response must be a mapping")
        allocation = snapshot.get("allocation")
        if not isinstance(allocation, Mapping):
            raise ValueError("status response has no allocation")
        release = allocation.get("controller_release")
        if not isinstance(release, str) or not release.strip() or release == "unknown":
            raise ValueError("status response has no known controller release")
        if release != SCRUFFY_COMMIT:
            raise ValueError(
                f"controller reports Scruffy {release}, expected {SCRUFFY_COMMIT}"
            )
    except Exception as exc:  # noqa: BLE001 - no admission on malformed status
        raise RuntimeError("Scruffy controller release attestation failed") from exc
    result = submit_scruffy_workflow(workflow, root=args.scruffy_root)
    print(json.dumps({"workflow": _describe(workflow, code_commit=code_commit), "submission": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
