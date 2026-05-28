from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

from ..core.hooks import rank0_only

__all__ = ["CSVLogger", "make_csv_hooks"]


class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str] | None = None):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.fieldnames = fieldnames
        self._writer = None
        self._file = None

    def _read_existing_rows(self) -> tuple[List[str], List[Dict[str, Any]]]:
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return [], []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or []), list(reader)

    @staticmethod
    def _merge_fieldnames(existing: List[str], keys: List[str]) -> List[str]:
        fieldnames = list(existing)
        if "step" in keys and "step" not in fieldnames:
            fieldnames.insert(0, "step")
        for key in sorted(k for k in keys if k != "step"):
            if key not in fieldnames:
                fieldnames.append(key)
        return fieldnames

    def _open_writer(self, fieldnames: List[str], *, mode: str = "a"):
        self._file = open(self.path, mode, newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        return self._writer

    def _ensure_writer(self, keys: List[str]):
        if self._writer is None:
            if self.fieldnames is not None:
                fieldnames = self.fieldnames
                exists = os.path.exists(self.path)
            else:
                existing_fieldnames, rows = self._read_existing_rows()
                fieldnames = self._merge_fieldnames(existing_fieldnames, keys)
                exists = bool(existing_fieldnames)
                if exists and fieldnames != existing_fieldnames:
                    writer = self._open_writer(fieldnames, mode="w")
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({key: row.get(key, None) for key in fieldnames})
                    if self._file is not None:
                        self._file.close()
                    self._writer = None
                    self._file = None
            self._open_writer(fieldnames)
            if not exists:
                self._writer.writeheader()
        elif self.fieldnames is None:
            extras = [key for key in keys if key not in self._writer.fieldnames]
            if extras:
                existing_fieldnames, rows = self._read_existing_rows()
                fieldnames = self._merge_fieldnames(existing_fieldnames, keys)
                if self._file is not None:
                    self._file.close()
                writer = self._open_writer(fieldnames, mode="w")
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, None) for key in fieldnames})

    def write(self, payload: Dict[str, Any]):
        keys = list(payload.keys())
        self._ensure_writer(keys)
        # Permit extra keys by filtering to known fieldnames
        row = {k: payload.get(k, None) for k in self._writer.fieldnames}
        self._writer.writerow(row)
        if self._file is not None:
            self._file.flush()


def make_csv_hooks(path: str) -> Dict[str, List]:
    logger = CSVLogger(path)

    def on_log(logs, ctx):
        logger.write(logs)

    def on_eval_end(metrics, ctx):
        payload = {"step": ctx.get("step"), **metrics}
        logger.write(payload)

    return {
        "on_log": [rank0_only(on_log)],
        "on_eval_end": [rank0_only(on_eval_end)],
    }
