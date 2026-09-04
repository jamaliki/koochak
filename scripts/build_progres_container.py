#!/usr/bin/env python3
"""Build the pinned Progres Apptainer image atomically."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file():
        subprocess.run(["/usr/bin/apptainer", "test", str(args.output)], check=True)
        print(f"reusing validated image {args.output}")
        return
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    try:
        subprocess.run([
            "/usr/bin/apptainer", "build", "--fakeroot", "--ignore-fakeroot-command",
            str(temporary), str(args.definition),
        ], check=True)
        subprocess.run(["/usr/bin/apptainer", "test", str(temporary)], check=True)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"built and validated {args.output}")


if __name__ == "__main__":
    main()
