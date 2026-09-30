"""Move collections of files between stores.

    python -m koochak.data archive SRC DEST [--groups groups.csv] [--dry-run]
    python -m koochak.data pull SRC DEST [--include 'features/*/1abc.npz']
    python -m koochak.data verify LOCATION [--deep]
    python -m koochak.data ls LOCATION [--include GLOB] [--long]

Locations are paths or ``scheme://`` URIs from the stores file
(``koochak.storage.stores_file``). Run transfers on compute nodes rather than
login nodes; progress goes to stderr and reports to stdout as JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from typing import Optional, Sequence

from ..storage.archive import DEFAULT_PACK_BYTES, archive, pull, verify
from ..storage.collection import load_collection
from ..storage.store import open_store
from ..utils.sizes import parse_size


def _progress(min_interval: float = 5.0):
    started = time.perf_counter()
    last = [0.0]

    def report(stage: str, done: int, total: int, size: int) -> None:
        now = time.perf_counter()
        if done == total or now - last[0] >= min_interval:
            last[0] = now
            rate = size / max(now - started, 1e-9) / 1e6
            print(
                f"[koochak.data] {stage} {done}/{total} {size / 1e9:.2f} GB {rate:.1f} MB/s",
                file=sys.stderr,
                flush=True,
            )

    return report


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stores", default=None, help="stores file (default: $KOOCHAK_STORES)")
    parser = argparse.ArgumentParser(prog="python -m koochak.data", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    packer = commands.add_parser("archive", parents=[common], help="pack a local tree into a collection")
    packer.add_argument("source", help="local directory (path or scheme:// URI)")
    packer.add_argument("target", help="collection location")
    packer.add_argument("--groups", default=None, help="CSV with columns path,group[,order]")
    packer.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    packer.add_argument("--layout", choices=("packed", "objects"), default="packed")
    packer.add_argument("--pack-bytes", type=parse_size, default=DEFAULT_PACK_BYTES)
    packer.add_argument(
        "--object-bytes",
        type=parse_size,
        default=None,
        help="files at least this large become standalone objects (default: from the target's profile)",
    )
    packer.add_argument("--streams", type=int, default=None)
    packer.add_argument("--read-streams", type=int, default=8)
    packer.add_argument("--pending-packs", type=int, default=8)
    packer.add_argument("--dry-run", action="store_true", help="plan only; write nothing")

    puller = commands.add_parser("pull", parents=[common], help="restore files from a collection")
    puller.add_argument("source", help="collection location")
    puller.add_argument("target", help="destination directory (path or scheme:// URI)")
    puller.add_argument("--include", action="append", default=[], metavar="GLOB")
    puller.add_argument("--streams", type=int, default=None)
    puller.add_argument("--no-metadata", action="store_true", help="do not restore mode and mtime")

    checker = commands.add_parser("verify", parents=[common], help="check a collection")
    checker.add_argument("location")
    checker.add_argument("--deep", action="store_true", help="re-read every byte and check SHA256")
    checker.add_argument("--streams", type=int, default=None)

    lister = commands.add_parser("ls", parents=[common], help="list a collection's files")
    lister.add_argument("location")
    lister.add_argument("--include", action="append", default=[], metavar="GLOB")
    lister.add_argument("--long", action="store_true", help="show size, digest, and location")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    stores = args.stores
    if args.command == "archive":
        report = archive(
            open_store(args.source, stores_file=stores),
            open_store(args.target, stores_file=stores),
            groups=args.groups,
            exclude=args.exclude,
            layout=args.layout,
            pack_bytes=args.pack_bytes,
            object_bytes=args.object_bytes,
            streams=args.streams,
            read_streams=args.read_streams,
            pending_packs=args.pending_packs,
            dry_run=args.dry_run,
            progress=_progress(),
        )
        print(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True))
        return 0
    if args.command == "pull":
        report = pull(
            open_store(args.source, stores_file=stores),
            open_store(args.target, stores_file=stores),
            include=args.include,
            streams=args.streams,
            restore_metadata=not args.no_metadata,
            progress=_progress(),
        )
        print(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        report = verify(open_store(args.location, stores_file=stores), deep=args.deep, streams=args.streams)
        print(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    collection = load_collection(open_store(args.location, stores_file=stores))
    for entry in collection.select(args.include):
        if not args.long:
            print(entry.path)
            continue
        where = entry.object if entry.object is not None else f"{collection.packs[entry.pack].key}@{entry.offset}"
        print(f"{entry.size:>14} {entry.sha256[:12]} {where} {entry.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
