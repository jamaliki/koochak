#!/usr/bin/env python3
"""Submit production-shaped resident-shard gates for mixture training."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    PreparedTask,
    PreparedWorkflow,
    prepare_run,
    submit_scruffy_workflow,
)
import scripts.submit_patch_coarse_mixture_unconditioned as unconditioned  # noqa: E402


GATE_STEPS = 600
GATE_RESOURCES = {
    **unconditioned.TRAIN_RESOURCES[128],
    "time_limit_seconds": 43_200,
}
PRODUCTION_CONFIGS = {
    "unconditioned": (
        unconditioned.REMOTE_RUN_ROOT
        / "patch-coarse-mixture-unconditioned-L128"
        / "a6c3b7d427f62231af0a17d41f96bf1fa925e671"
        / "train/L128/pool_before_attention_pair_transition-mix75_25-sc0p5/config.yaml"
    ),
    "progres": (
        unconditioned.REMOTE_RUN_ROOT
        / "patch-coarse-mixture-progres-conditioned-L128"
        / "96f0e3f970da4bd7a9fb1a74b76eff55f9a440b1"
        / "train/L128/pool_before_attention_pair_transition-mix75_25-sc0p5-progres/config.yaml"
    ),
}
ALLOWED_DIFFS = {
    "data.shard_cache_size",
    "logging.csv_path",
    "logging.jsonl_path",
    "train.ckpt_every",
    "train.eval_at_step_zero",
    "train.eval_every",
    "train.log_every",
    "train.max_steps",
    "train.out_dir",
    "train.save_final",
    "wandb.enabled",
    "wandb.mode",
}


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        return {
            child_key: child_value
            for key, child in value.items()
            for child_key, child_value in _flatten(
                child, f"{prefix}.{key}" if prefix else str(key)
            ).items()
        }
    return {prefix: value}


def resolved_diff(parent: Mapping[str, object], child: Mapping[str, object]) -> list[dict[str, object]]:
    """Return and validate the exact immutable parent-to-gate diff."""

    parent_flat = _flatten(parent)
    child_flat = _flatten(child)
    differences = [
        {"path": key, "parent": parent_flat.get(key), "child": child_flat.get(key)}
        for key in sorted(set(parent_flat) | set(child_flat))
        if parent_flat.get(key) != child_flat.get(key)
    ]
    observed = {item["path"] for item in differences}
    if observed != ALLOWED_DIFFS:
        raise AssertionError(
            f"unexpected resident-gate config differences: "
            f"observed={sorted(observed)}, expected={sorted(ALLOWED_DIFFS)}"
        )
    if child_flat["data.shard_cache_size"] is not None:
        raise AssertionError("resident gate must use data.shard_cache_size=null")
    return differences


def _config(prepared) -> dict[str, object]:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    value = OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config must be a mapping")
    return value


def _patches(run_dir: Path) -> list[ConfigPatch]:
    return [
        ConfigPatch("data.shard_cache_size", None),
        ConfigPatch("train.max_steps", GATE_STEPS),
        ConfigPatch("train.log_every", 1),
        ConfigPatch("train.eval_every", 1_000_000_000),
        ConfigPatch("train.eval_at_step_zero", False),
        ConfigPatch("train.ckpt_every", 1_000_000_000),
        ConfigPatch("train.save_final", False),
        ConfigPatch("logging.csv_path", str(run_dir / "log.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "log.jsonl")),
        ConfigPatch("wandb.enabled", False),
        ConfigPatch("wandb.mode", "disabled"),
    ]


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, dict[str, object]]:
    short = code_commit[:7]
    workflow_id = f"hk-patch-coarse-mixture-resident-gate-L128-{short}"
    output_root = unconditioned.REMOTE_RUN_ROOT / "patch-coarse-mixture-resident-gate-L128" / code_commit
    remote_cwd = unconditioned.REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    profile = unconditioned._load_profile(unconditioned.GPU_PROFILE)
    tasks = []
    diffs = {}
    for label, parent_config in PRODUCTION_CONFIGS.items():
        run_dir = output_root / label
        patches = _patches(run_dir)
        prepared = prepare_run(
            name=f"hk-mixture-resident-gate-{label}-{short}",
            profile=profile,
            python_args=[
                "-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"
            ],
            cwd=str(remote_cwd),
            run_dir=str(run_dir),
            base_config=parent_config,
            patches=patches,
        )
        child = _config(prepared)
        parent = OmegaConf.to_container(OmegaConf.load(parent_config), resolve=True)
        if not isinstance(parent, dict):
            raise TypeError("parent config must be a mapping")
        diffs[label] = {
            "parent_config": str(parent_config),
            "resolved_differences": resolved_diff(parent, child),
        }
        tasks.append(
            PreparedTask(
                task_id=f"gate-{label}",
                run=prepared,
                resources=GATE_RESOURCES,
                recovery=unconditioned.RECOVERY,
            )
        )
    workflow = PreparedWorkflow(
        request_id=f"{unconditioned.PROJECT_ID}/{workflow_id}/v1",
        workflow_id=workflow_id,
        project_id=unconditioned.PROJECT_ID,
        tasks=tuple(tasks),
    )
    return workflow, {
        "workflow_id": workflow_id,
        "request_id": workflow.request_id,
        "code_commit": code_commit,
        "gate_steps": GATE_STEPS,
        "cache_policy": "resident_unique_owned_shards",
        "shard_cache_size": None,
        "tasks": diffs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = unconditioned._git("rev-parse", "HEAD")
    if any(not file.is_file() for file in PRODUCTION_CONFIGS.values()):
        if not args.dry_run:
            raise FileNotFoundError("one or more immutable production configs are missing")
    workflow, description = build_workflow(code_commit)
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True))
        return
    unconditioned._validate_online(code_commit)
    sys.path.insert(0, str(unconditioned.SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(unconditioned.SCRUFFY_ROOT)
    attestation = unconditioned._validate_scruffy_snapshot(
        snapshot,
        expected_allocation_id=os.environ.get("SCRUFFY_ALLOCATION_ID"),
    )
    if attestation["controller_release"] != unconditioned.SCRUFFY_COMMIT:
        raise RuntimeError("Scruffy controller release mismatch")
    output_root = unconditioned.REMOTE_RUN_ROOT / "patch-coarse-mixture-resident-gate-L128" / code_commit
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config_diffs.json").write_text(
        json.dumps(description, indent=2, sort_keys=True) + "\n"
    )
    submission = submit_scruffy_workflow(workflow, root=unconditioned.SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "allocation": attestation, "submission": submission}, indent=2, default=str))


if __name__ == "__main__":
    main()
