#!/usr/bin/env python3
"""Continue one baseline from an explicitly verified Koochak checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

import torch  # noqa: E402

from hierarchical_kaveh.config import RunConfig, load_config  # noqa: E402
from hierarchical_kaveh.training import run_training  # noqa: E402


_PUBLICATION_KEYS = {"v", "artifact_id", "path", "size_bytes", "sha256", "manifest_path"}


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(file: Path, label: str) -> None:
    if file.is_symlink() or not file.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {file}")


def validate_source_checkpoint(source: Path, *, expected_step: int) -> dict[str, Any]:
    """Validate immutable publication evidence and the serialized state cursor."""

    expected_name = f"step{expected_step:09d}.pt"
    if source.name != expected_name:
        raise ValueError(f"source checkpoint must be named {expected_name}, got {source.name}")
    source = source.absolute()
    manifest = Path(f"{source}.ready.json")
    _regular_file(source, "source checkpoint")
    _regular_file(manifest, "source checkpoint publication")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or set(record) != _PUBLICATION_KEYS:
        raise RuntimeError(f"source publication schema mismatch: {manifest}")
    expected_manifest = str(manifest.absolute())
    if (
        record["v"] != 1
        or record["artifact_id"] != f"checkpoint/{expected_name}"
        or record["path"] != str(source)
        or record["manifest_path"] != expected_manifest
    ):
        raise RuntimeError(f"source publication identity mismatch: {manifest}")
    size = source.stat().st_size
    if type(record["size_bytes"]) is not int or record["size_bytes"] != size:
        raise RuntimeError(f"source publication size mismatch: {source}")
    digest = _sha256(source)
    if record["sha256"] != digest:
        raise RuntimeError(f"source checkpoint SHA-256 mismatch: {source}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("source checkpoint payload is not a mapping")
    if checkpoint.get("step") != expected_step:
        raise RuntimeError(
            f"source checkpoint cursor {checkpoint.get('step')!r} differs from {expected_step}"
        )
    next_step = checkpoint.get("next_step")
    if type(next_step) is not int or next_step not in {expected_step, expected_step + 1}:
        raise RuntimeError(f"source checkpoint has invalid next_step: {next_step!r}")
    if not isinstance(checkpoint.get("model"), Mapping) or not checkpoint["model"]:
        raise RuntimeError("source checkpoint has no model state")
    if not isinstance(checkpoint.get("optimizer"), Mapping) or not checkpoint["optimizer"]:
        raise RuntimeError("source checkpoint has no optimizer state")
    return {
        "path": str(source),
        "manifest_path": expected_manifest,
        "artifact_id": record["artifact_id"],
        "size_bytes": size,
        "sha256": digest,
        "step": expected_step,
        "next_step": next_step,
        "model_parameter_entries": len(checkpoint["model"]),
        "optimizer_state_entries": len(checkpoint["optimizer"]),
    }


def _own_resume(config: RunConfig) -> tuple[str, dict[str, Any] | None]:
    """Select only a validated checkpoint from this recovery run's own output."""

    output = Path(config.train.out_dir)
    numbered = sorted(output.glob("step*.pt")) if output.is_dir() else []
    selection = None
    if numbered:
        from koochak.storage import checkpoint as checkpoint_lib

        selection = checkpoint_lib.resolve_auto_resume(str(output))
        if selection is None:
            raise RuntimeError(f"recovery output contains no valid published checkpoint: {output}")
    if selection is not None:
        return "auto", {"path": selection[0], "next_step": selection[1].get("next_step")}
    latest = output / "latest.pt"
    if latest.exists() or latest.is_symlink():
        raise RuntimeError(f"refusing ambiguous recovery output pointer: {latest}")
    return "source", None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-step", type=int, default=450_000)
    parser.add_argument("--target-step", type=int, default=500_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source_step < 0 or args.target_step <= args.source_step:
        raise ValueError("target-step must be greater than source-step")
    config = load_config(args.config)
    if config.train.max_steps != args.target_step:
        raise ValueError("recovery config max_steps must equal target-step")
    source_evidence = validate_source_checkpoint(
        args.source_checkpoint, expected_step=args.source_step
    )
    resume_kind, own_evidence = _own_resume(config)
    if own_evidence is None:
        resume: str | Path = args.source_checkpoint.absolute()
    else:
        resume = "auto"
    print(
        json.dumps(
            {
                "source": source_evidence,
                "resume_selection": resume_kind,
                "own_checkpoint": own_evidence,
                "output_dir": config.train.out_dir,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    run_training(config, resume=resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
