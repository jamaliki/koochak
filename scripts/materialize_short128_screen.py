#!/usr/bin/env python3
"""Materialize the controlled eight-config short-128 training screen."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from hierarchical_kaveh.config import load_config


VARIANTS = (
    ("lr1e3_none", 1.0e-3, 0.0, 0.0),
    ("lr1e3_lddt", 1.0e-3, 1.0, 0.0),
    ("lr1e3_dist", 1.0e-3, 0.0, 0.5),
    ("lr1e3_both", 1.0e-3, 1.0, 0.5),
    ("lr3e4_none", 3.0e-4, 0.0, 0.0),
    ("lr3e4_lddt", 3.0e-4, 1.0, 0.0),
    ("lr3e4_dist", 3.0e-4, 0.0, 0.5),
    ("lr3e4_both", 3.0e-4, 1.0, 0.5),
)


def materialize(
    base_config: Path,
    output_root: Path,
    metadata_file: Path,
    *,
    preflight: bool,
) -> list[dict[str, object]]:
    common = load_config(base_config).to_dict()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant, learning_rate, lddt_weight, distogram_weight in VARIANTS:
        values = deepcopy(common)
        run_dir = output_root / variant
        run_dir.mkdir(parents=True, exist_ok=True)
        values["data"]["metadata_path"] = str(metadata_file)
        values["optimizer"]["lr"] = learning_rate
        values["loss"]["smooth_lddt_weight"] = lddt_weight
        values["loss"]["distogram_weight"] = distogram_weight
        values["train"]["out_dir"] = str(run_dir)
        values["logging"]["csv_path"] = str(run_dir / "log.csv")
        values["logging"]["jsonl_path"] = str(run_dir / "log.jsonl")
        values["wandb"]["enabled"] = not preflight
        values["wandb"]["mode"] = None if preflight else "online"
        values["wandb"]["name"] = f"short128-{variant}"
        values["wandb"]["tags"] = [
            "short128",
            "3a3r8c",
            f"lr={learning_rate:g}",
            f"lddt={lddt_weight:g}",
            f"distogram={distogram_weight:g}",
        ]
        if preflight:
            values["data"]["num_workers"] = 0
            values["data"]["persistent_workers"] = False
            values["train"]["max_steps"] = 2
            values["train"]["log_every"] = 1
            values["train"]["ckpt_every"] = 1
            values["train"]["keep_last_k"] = 2
        config_file = run_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(values), config_file)
        digest = hashlib.sha256(config_file.read_bytes()).hexdigest()
        manifest.append(
            {
                "variant": variant,
                "config": str(config_file),
                "sha256": digest,
                "learning_rate": learning_rate,
                "smooth_lddt_weight": lddt_weight,
                "distogram_weight": distogram_weight,
                "preflight": preflight,
            }
        )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/experiments/short128_100k.yaml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = materialize(
        args.base_config,
        args.output_root,
        args.metadata,
        preflight=args.preflight,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

