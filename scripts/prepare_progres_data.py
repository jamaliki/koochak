#!/usr/bin/env python3
"""Download and validate the immutable Progres model/database bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import progres


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    progres.download_data_if_required()
    files = sorted(file for file in Path(progres.data_dir).rglob("*") if file.is_file())
    model = Path(progres.trained_model_fp)
    if not model.is_file() or not Path(f"{model}.okay").is_file():
        raise RuntimeError(f"Progres model validation marker is missing: {model}")
    result = {
        "progres_version": importlib.metadata.version("progres"),
        "data_dir": str(Path(progres.data_dir).resolve()),
        "file_count": len(files),
        "total_bytes": sum(file.stat().st_size for file in files),
        "trained_model": str(model),
        "trained_model_sha256": _sha256(model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
