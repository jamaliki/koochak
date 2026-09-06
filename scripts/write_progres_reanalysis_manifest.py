#!/usr/bin/env python3
"""Publish a fail-closed availability manifest for Progres reanalysis inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def build_manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    """Check declared inputs and retain missing entries as explicit records."""

    seen: set[str] = set()
    observed = []
    for original in entries:
        panel_id = str(original["panel_id"])
        if panel_id in seen:
            raise ValueError(f"duplicate panel ID: {panel_id}")
        seen.add(panel_id)
        required = [Path(str(item)) for item in original.get("required_paths", [])]
        missing = [str(item) for item in required if not item.exists()]
        observed.append(
            {
                **original,
                "availability_status": "available" if not missing else "missing",
                "missing_paths": missing,
            }
        )
    return {
        "schema": "progres-complete-linkage-availability-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(observed),
        "available_count": sum(item["availability_status"] == "available" for item in observed),
        "missing_count": sum(item["availability_status"] == "missing" for item in observed),
        "entries": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.entries_json)
    if not isinstance(entries, list):
        raise ValueError("--entries-json must encode a list")
    result = build_manifest(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "entry_count": result["entry_count"],
        "available_count": result["available_count"],
        "missing_count": result["missing_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
