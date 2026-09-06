#!/usr/bin/env python3
"""Attest the immutable Progres v1.1.0 weights before downstream analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/progres-data/v1.1.0")
FILES = {
    "trained_model.pt": {
        "md5": "c490293eb8d0bb350e68a8229c6884da",
        "url": "https://zenodo.org/api/records/18245422/files/trained_model.pt/content",
    },
    "cath40.pt": {
        "md5": "616510a3b21d45500b26dd5ae9a040d6",
        "url": "https://zenodo.org/api/records/18245422/files/cath40.pt/content",
    },
}
CHUNK_BYTES = 8 * 1024 * 1024


def _md5(file: Path) -> str:
    digest = hashlib.md5()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest(data_dir: Path) -> dict[str, object]:
    if data_dir.is_symlink():
        raise RuntimeError(f"Progres root must not be a symlink: {data_dir}")
    resolved_dir = data_dir.resolve()
    if resolved_dir != EXPECTED_ROOT:
        raise RuntimeError(
            f"refusing unpinned Progres root: expected {EXPECTED_ROOT}, got {resolved_dir}"
        )
    if not resolved_dir.is_dir():
        raise RuntimeError(f"Progres root is not a regular directory: {resolved_dir}")

    observed: dict[str, object] = {}
    for name, expected in FILES.items():
        file = resolved_dir / name
        if file.is_symlink() or not file.is_file():
            raise RuntimeError(f"Progres artifact is not a regular file: {file}")
        digest = _md5(file)
        if digest != expected["md5"]:
            raise RuntimeError(
                f"Progres artifact digest mismatch for {file}: "
                f"expected {expected['md5']}, got {digest}"
            )
        observed[name] = {
            "path": str(file),
            "bytes": file.stat().st_size,
            "md5": digest,
            "expected_md5": expected["md5"],
            "url": expected["url"],
        }
    return {
        "schema": "hk-progres-data-attestation-v1",
        "progres_version": "1.1.0",
        "zenodo_record": "18245422",
        "data_dir": str(resolved_dir),
        "files": observed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = attest(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
