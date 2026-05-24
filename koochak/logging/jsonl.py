from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TextIO

from ..core.hooks import rank0_only

__all__ = ["JSONLLogger", "make_jsonl_hooks"]


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._file: Optional[TextIO] = open(self.path, "a", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def make_jsonl_hooks(path: str) -> Dict[str, List]:
    logger = JSONLLogger(path)

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
