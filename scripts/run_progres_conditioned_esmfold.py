#!/usr/bin/env python3
"""Refold the paired null/conditioned panel and emit one typed score artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_esmfold_designability_shard import _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    sample_manifest = json.loads((args.sample_root / "manifest.json").read_text(encoding="utf-8"))
    if sample_manifest.get("schema") != "progres-paired-samples-v1":
        raise ValueError("paired sample manifest schema mismatch")
    groups = sample_manifest.get("groups")
    if not isinstance(groups, list) or len(groups) != 8:
        raise ValueError("paired sample manifest must contain eight groups")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat = args.output_dir / "inputs"
    if flat.exists():
        shutil.rmtree(flat)
    flat.mkdir()
    mapping: dict[str, dict[str, object]] = {}
    index = 0
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("paired sample group is malformed")
        for mode, key in (("null", "null_dir"), ("conditioned", "conditioned_dir")):
            source = Path(str(group[key]))
            fastas = sorted(source.glob("sample_*.fasta"))
            if len(fastas) != 4:
                raise ValueError(f"expected four {mode} samples in {source}")
            for replicate, fasta in enumerate(fastas):
                identifier = f"sample_{index:06d}"
                shutil.copy2(fasta, flat / f"{identifier}.fasta")
                shutil.copy2(fasta.with_suffix(".pdb"), flat / f"{identifier}.pdb")
                mapping[identifier] = {
                    "condition_mode": mode,
                    "target_rank": int(group["target_rank"]),
                    "target_id": str(group["target_id"]),
                    "replicate": replicate,
                    "paired_seed": int(group["seed"]),
                }
                index += 1
    if index != 64:
        raise ValueError(f"expected 64 paired samples, found {index}")
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_esmfold_designability_shard.py")),
        "--sample-dir", str(flat),
        "--output-dir", str(args.output_dir / "refolded"),
        "--variant", "progres-paired",
        "--step", "0",
        "--length", "128",
        "--expected-count", "64",
        "--wrapper", str(args.wrapper),
        "--chunk-size", str(args.chunk_size),
    ]
    if args.bf16:
        command.append("--bf16")
    subprocess.run(command, check=True)
    rows = json.loads((args.output_dir / "refolded" / "per_sample.json").read_text(encoding="utf-8"))
    if {str(row["id"]) for row in rows} != set(mapping):
        raise ValueError("ESMFold rows do not match paired sample identities")
    for row in rows:
        row.update(mapping[str(row["id"])])
        row["paired_noise_contract"] = True
    rows.sort(key=lambda row: str(row["id"]))
    (args.output_dir / "per_sample.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "progres-paired-esmfold-v1",
        "length": 128,
        "sample_count": len(rows),
        "null_count": sum(row["condition_mode"] == "null" for row in rows),
        "conditioned_count": sum(row["condition_mode"] == "conditioned" for row in rows),
        "paired_noise_contract": True,
        "metrics": _summary(rows),
        "source_manifest": str((args.sample_root / "manifest.json").resolve()),
        "command": command,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
