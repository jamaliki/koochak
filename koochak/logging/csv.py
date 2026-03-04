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

    def _ensure_writer(self, keys: List[str]):
        if self._writer is None:
            # Decide header
            fieldnames = self.fieldnames or ["step"] + sorted(k for k in keys if k != "step")
            exists = os.path.exists(self.path)
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            if not exists:
                self._writer.writeheader()

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
