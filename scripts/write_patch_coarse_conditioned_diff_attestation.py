#!/usr/bin/env python3
"""Materialize the immutable conditioned-factorial config diff attestation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import submit_patch_coarse_mixture_conditioned as conditioned  # noqa: E402


PARENT_COMMIT = conditioned.PARENT_COMMIT
CHILD_COMMIT = "96f0e3f970da4bd7a9fb1a74b76eff55f9a440b1"
CHILD_ROOT = (
    conditioned.REMOTE_RUN_ROOT
    / "patch-coarse-mixture-progres-conditioned-L128"
    / CHILD_COMMIT
)


def _child_config(cell: conditioned.Cell) -> Path:
    return CHILD_ROOT / "train" / "L128" / f"{cell.cell_id}" / "config.yaml"


def build_attestation() -> dict[str, object]:
    diffs = []
    for cell in conditioned.CELLS:
        parent = conditioned.PARENT_CELLS[(cell.architecture, cell.mixture_id)]
        child = _child_config(cell)
        if not parent.is_file():
            raise FileNotFoundError(f"immutable parent config is missing: {parent}")
        if not child.is_file():
            raise FileNotFoundError(f"immutable conditioned config is missing: {child}")
        resolved_child = OmegaConf.to_container(OmegaConf.load(child), resolve=True)
        diff = conditioned.resolved_diff(
            parent, resolved_child, cell, target_cache_size=8
        )
        if {item["classification"] for item in diff["differences"]} - {
            "conditioning",
            "run_identity_or_output",
        }:
            raise AssertionError(f"unexpected config diff classification for {cell.cell_id}")
        diffs.append({**diff, "child_config": str(child), "child_config_sha256": conditioned.hashlib.sha256(child.read_bytes()).hexdigest()})
    return {
        "schema": "hk-patch-coarse-conditioned-resolved-config-attestation-v1",
        "parent_commit": PARENT_COMMIT,
        "parent_workflow": conditioned.PARENT_WORKFLOW,
        "child_commit": CHILD_COMMIT,
        "child_root": str(CHILD_ROOT),
        "allowed_paths": sorted(conditioned.CONDITION_PATHS | conditioned.OUTPUT_PATHS),
        "required_operational_invariants": {
            "data.shard_cache_size": 8,
            "mixture_inherited_verbatim": True,
        },
        "cells": diffs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_attestation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "cells": len(document["cells"])}, sort_keys=True))


if __name__ == "__main__":
    main()
