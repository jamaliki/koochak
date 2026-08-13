#!/usr/bin/env python3
"""Materialize a lossless resident-shard resume config for one training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from hierarchical_kaveh.config import load_config


def materialize(source: Path, output: Path) -> None:
    """Copy a strict run config and change only loader/W&B resume policy."""

    values = load_config(source).to_dict()
    values["data"]["shard_cache_size"] = None
    values["wandb"]["resume"] = "must"
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(values), output)
    load_config(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
