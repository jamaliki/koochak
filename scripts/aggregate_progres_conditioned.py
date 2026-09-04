#!/usr/bin/env python3
"""Collect the four conditional-cell milestone analyses into one artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="cell_id=analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    cells: dict[str, object] = {}
    for item in args.input:
        cell_id, separator, filename = item.partition("=")
        if not separator or not cell_id or not filename or cell_id in cells:
            raise ValueError("each --input must be a unique cell_id=analysis.json")
        value = json.loads(Path(filename).read_text(encoding="utf-8"))
        if value.get("schema") != "progres-paired-analysis-v1":
            raise ValueError(f"analysis schema mismatch for {cell_id}")
        cells[cell_id] = value
    if len(cells) != 4:
        raise ValueError("aggregate requires exactly four conditional cells")
    result = {"schema": "progres-paired-factorial-v1", "step": args.step, "cell_count": len(cells), "cells": cells}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cell_count": len(cells), "step": args.step}, sort_keys=True))


if __name__ == "__main__":
    main()
