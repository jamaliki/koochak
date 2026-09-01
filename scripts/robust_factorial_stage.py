#!/usr/bin/env python3
"""Run a factorial downstream stage and publish one strict artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from koochak.storage.artifact import DeclaredOutput, publish_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sample", "esmfold", "analysis"), required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--kind", choices=("file", "directory"), required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _validate(stage: str, output: Path, expected_records: int | None) -> int | None:
    if stage == "sample":
        manifest = output / "manifest.json"
        if not output.is_dir() or not manifest.is_file():
            raise RuntimeError("sample output is missing manifest.json")
        observed = len(tuple(output.rglob("*.fasta")))
        if expected_records is not None and observed != expected_records:
            raise RuntimeError(f"sample produced {observed} FASTA files, expected {expected_records}")
        return observed
    if stage == "esmfold":
        required = ("summary.json", "per_sample.json", "per_sample.csv")
        if not output.is_dir() or any(not (output / name).is_file() for name in required):
            raise RuntimeError("ESMFold output is missing a required summary file")
        if not tuple(output.rglob("*.pdb")):
            raise RuntimeError("ESMFold output contains no PDB files")
        return expected_records
    if output != output.resolve() or not output.is_file():
        raise RuntimeError("analysis output is not a regular file")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("a stage command is required")
    subprocess.run([sys.executable, *command], check=True)
    observed = _validate(args.stage, args.artifact_path, args.expected_records)
    output = DeclaredOutput(
        args.artifact_id,
        str(args.artifact_path),
        kind=args.kind,
        stage=args.stage,
        provenance={
            "project_id": args.project,
            "workflow_id": args.workflow,
            "task_id": args.task,
            "code_commit": args.code_commit,
        },
        expected_records=args.expected_records,
        metadata={"stage": args.stage, "workflow": args.workflow},
    )
    publish_artifact(output, observed_records=observed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
