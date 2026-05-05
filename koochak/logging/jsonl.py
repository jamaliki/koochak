from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ..core.hooks import rank0_only

__all__ = ["JSONLLogger", "make_jsonl_hooks"]


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        self._file.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def make_jsonl_hooks(path: str) -> Dict[str, List]:
    logger = JSONLLogger(path)

    def on_log(logs, ctx):
        logger.write(logs)

    def on_eval_end(metrics, ctx):
        payload = {"step": ctx.get("step"), **metrics}
        logger.write(payload)

    def on_train_end(ctx):
        logger.close()

    return {
        "on_log": [rank0_only(on_log)],
        "on_eval_end": [rank0_only(on_eval_end)],
        "on_train_end": [rank0_only(on_train_end)],
    }
