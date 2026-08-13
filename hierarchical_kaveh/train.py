"""Command-line entry point for Pallatom-aligned monomer EDM training."""

from __future__ import annotations

import argparse

from .config import load_config
from .training import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Strict YAML run configuration.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint file, or 'latest' to resume from train.out_dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    run_training(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
