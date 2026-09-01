#!/usr/bin/env python3
"""Run one small canary stage and publish its strict Koochak artifact.

The sampler delegates to the project's scientific sampler.  Fold and
analysis are intentionally tiny, deterministic checks of the preceding
artifact; keeping these adapters here gives every non-training stage a real
declared-output manifest without adding scheduler state to the project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from koochak.storage.artifact import DeclaredOutput, publish_artifact


def _provenance(args: argparse.Namespace) -> dict[str, str]:
    return {
        "project_id": args.project,
        "workflow_id": args.workflow,
        "task_id": args.task,
        "code_commit": args.code_commit,
    }


def _publish(args: argparse.Namespace, *, observed_records: int) -> None:
    output = DeclaredOutput(
        args.artifact_id,
        str(args.artifact_path),
        kind=args.kind,
        stage=args.stage,
        provenance=_provenance(args),
        expected_records=args.expected_records,
        metadata={"canary": True, "stage": args.stage},
    )
    publish_artifact(output, observed_records=observed_records)


def _sample(args: argparse.Namespace) -> int:
    args.artifact_path.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "scripts.sample_short128_milestone",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.artifact_path),
        "--lengths",
        args.lengths,
        "--samples-per-length",
        str(args.samples_per_length),
        "--batch-size",
        str(args.samples_per_length),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--precision",
        args.precision,
    ]
    subprocess.run(command, check=True)
    count = sum(1 for _ in args.artifact_path.glob("L*/sample_*.fasta"))
    if count != args.expected_records:
        raise RuntimeError(f"sampler produced {count} records, expected {args.expected_records}")
    _publish(args, observed_records=count)
    return 0


def _fold(args: argparse.Namespace) -> int:
    source = args.input_path / "manifest.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    source_manifest = json.loads(source.read_text(encoding="utf-8"))
    result = {
        "v": 1,
        "stage": "fold",
        "sample_manifest": str(source.resolve()),
        "sample_count": sum(
            1 for _ in args.input_path.glob("L*/sample_*.fasta")
        ),
        "status": "canary-folded",
        "source_checkpoint_step": source_manifest.get("checkpoint_step"),
    }
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _publish(args, observed_records=1)
    return 0


def _analysis(args: argparse.Namespace) -> int:
    fold = json.loads(args.input_path.read_text(encoding="utf-8"))
    result = {
        "v": 1,
        "stage": "analysis",
        "fold_status": fold.get("status"),
        "sample_count": fold.get("sample_count"),
        "status": "canary-analyzed",
    }
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _publish(args, observed_records=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sample", "fold", "analysis"), required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--kind", choices=("file", "directory"), required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--lengths", default="8")
    parser.add_argument("--samples-per-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args(argv)
    if args.expected_records < 0:
        raise ValueError("expected-records must be non-negative")
    if args.stage == "sample":
        if args.kind != "directory" or args.config is None or args.checkpoint is None:
            raise ValueError("sample requires a directory artifact, config, and checkpoint")
        return _sample(args)
    if args.input_path is None or args.kind != "file":
        raise ValueError(f"{args.stage} requires an input path and file artifact")
    return _fold(args) if args.stage == "fold" else _analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
