#!/usr/bin/env python3
"""Run a milestone sampler as soon as its checkpoint becomes available."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=172800.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("sample_script", type=Path)
    parser.add_argument("sample_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError("timeout-seconds and poll-seconds must be positive")
    deadline = time.monotonic() + args.timeout_seconds
    while not args.checkpoint.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"checkpoint did not appear: {args.checkpoint}")
        time.sleep(args.poll_seconds)
    command = [sys.executable, str(args.sample_script), *args.sample_args]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
