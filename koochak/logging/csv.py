from __future__ import annotations

import csv
import os
import warnings
from typing import Any, Dict, List, Optional, TextIO

from ..core.hooks import rank0_only

__all__ = ["CSVLogger", "make_csv_hooks"]


class CSVLogger:
    """Append-only CSV writer with a stable column set.

    The header is determined the first time `write` is called. Subsequent payloads
    are mapped against that header — extra keys are reported once via a `UserWarning`
    so the operator notices schema drift rather than having data silently dropped.
    """

    def __init__(self, path: str, fieldnames: Optional[List[str]] = None) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.fieldnames: Optional[List[str]] = list(fieldnames) if fieldnames else None
        self._writer: Optional[csv.DictWriter] = None
        self._file: Optional[TextIO] = None
        self._missing_warned: set[str] = set()

    def _ensure_writer(self, keys: List[str]) -> None:
        if self._writer is not None:
            return
        fieldnames = self.fieldnames or (["step"] + sorted(k for k in keys if k != "step"))
        self.fieldnames = list(fieldnames)
        exists = os.path.exists(self.path)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if not exists:
            self._writer.writeheader()

    def write(self, payload: Dict[str, Any]) -> None:
        self._ensure_writer(list(payload.keys()))
        assert self._writer is not None and self._file is not None and self.fieldnames is not None
        extras = set(payload.keys()) - set(self.fieldnames)
        new_extras = extras - self._missing_warned
        if new_extras:
            warnings.warn(
                f"CSVLogger: dropping keys not in initial schema: {sorted(new_extras)}",
                UserWarning,
            )
            self._missing_warned |= new_extras
        row = {k: payload.get(k, None) for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def make_csv_hooks(path: str) -> Dict[str, List]:
    logger = CSVLogger(path)

    def on_log(logs, ctx):
        logger.write(logs)

    def on_eval_end(metrics, ctx):
        payload = {"step": ctx.get("step"), **metrics}
        logger.write(payload)

    def on_close(*_args, **_kwargs):
        logger.close()

    return {
        "on_log": [rank0_only(on_log)],
        "on_eval_end": [rank0_only(on_eval_end)],
        "on_train_end": [rank0_only(on_close)],
        "on_exception": [rank0_only(on_close)],
    }
