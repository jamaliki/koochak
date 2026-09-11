#!/usr/bin/env python3
"""Run a historical sampler with current evaluation tooling.

Historical model commits intentionally predate the shared evaluation scripts
and may reject newer, irrelevant config fields. This wrapper keeps the
historical model package and sampler, filters only fields unknown to that
commit's dataclasses, and leaves downstream ESMFold/Progres evaluation to the
current evaluator checkout.
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING, fields, is_dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from omegaconf import OmegaConf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--samples-per-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--raw", action="store_true")
    return parser


def _field_defaults(cls: type[Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for item in fields(cls):
        if item.default_factory is not MISSING:
            defaults[item.name] = item.default_factory()
        elif item.default is not MISSING:
            defaults[item.name] = item.default
    return defaults


def _filter_section(cls: type[Any], value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    defaults = _field_defaults(cls)
    filtered: dict[str, Any] = {}
    for name, child in value.items():
        if name not in allowed:
            continue
        default = defaults.get(name)
        if is_dataclass(default) and isinstance(child, Mapping):
            filtered[name] = _filter_section(type(default), child)
        else:
            filtered[name] = child
    return filtered


def _sanitized_config(source_root: Path, config_path: Path, output_dir: Path) -> Path:
    sys.path.insert(0, str(source_root))
    from hierarchical_kaveh.config import RunConfig  # noqa: PLC0415

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    sanitized = _filter_section(RunConfig, raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="historical-config-", dir=output_dir,
        delete=False, encoding="utf-8",
    )
    try:
        OmegaConf.save(config=OmegaConf.create(sanitized), f=handle.name)
    finally:
        handle.close()
    return Path(handle.name)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_root.resolve()
    sampler = source_root / "scripts" / "sample_short128_milestone.py"
    if not sampler.is_file():
        raise FileNotFoundError(sampler)
    sanitized = _sanitized_config(source_root, args.config, args.output_dir)
    command = [
        sys.executable, str(sampler),
        "--config", str(sanitized),
        "--checkpoint", str(args.checkpoint),
        "--output-dir", str(args.output_dir),
        "--lengths", args.lengths,
        "--samples-per-length", str(args.samples_per_length),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--precision", args.precision,
        "--allow-checkpoint-config-mismatch",
    ]
    if args.compile:
        command.append("--compile")
    if args.raw:
        command.append("--raw")
    env = os.environ.copy()
    pythonpath = [str(source_root), str(source_root / "external" / "koochak")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    try:
        subprocess.run(command, check=True, env=env)
    finally:
        sanitized.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
