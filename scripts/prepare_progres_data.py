#!/usr/bin/env python3
"""Download and validate the Progres model and CATH40 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import torch

FILES = {
    "trained_model.pt": (
        "https://zenodo.org/api/records/18245422/files/trained_model.pt/content",
        "c490293eb8d0bb350e68a8229c6884da",
    ),
    "cath40.pt": (
        "https://zenodo.org/api/records/18245422/files/cath40.pt/content",
        "616510a3b21d45500b26dd5ae9a040d6",
    ),
}


def _download(file: Path, url: str, expected_md5: str) -> None:
    if file.is_file() and hashlib.md5(file.read_bytes()).hexdigest() == expected_md5:
        return
    temporary = file.with_suffix(file.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    urllib.request.urlretrieve(url, temporary)
    if hashlib.md5(temporary.read_bytes()).hexdigest() != expected_md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"published digest mismatch for {file.name}")
    temporary.replace(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        _download(args.data_dir / name, url, digest)
    model = torch.load(args.data_dir / "trained_model.pt", map_location="cpu", weights_only=True)
    cath = torch.load(args.data_dir / "cath40.pt", map_location="cpu", weights_only=True)
    if "model" not in model or not {"ids", "embeddings", "nres", "notes"} <= cath.keys():
        raise RuntimeError("Progres artifacts have unexpected schemas")
    result = {
        "progres_version": "1.1.0",
        "zenodo_record": "18245422",
        "data_dir": str(args.data_dir.resolve()),
        "files": {
            name: {"bytes": (args.data_dir / name).stat().st_size, "md5": digest, "url": url}
            for name, (url, digest) in FILES.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
