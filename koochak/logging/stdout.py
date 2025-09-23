from __future__ import annotations

import math
import sys
from typing import Any, Dict, Iterable, List, Optional
from kaveh.koochak.core.hooks import rank0_only

__all__ = ["StdoutLogger", "make_stdout_hooks", "log_step_tsv"]


def _to_scalar(x: Any) -> Any:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            return x.detach().float().mean().item()
    except Exception:
        pass
    return x


def _fmt_val(v: Any) -> str:
    v = _to_scalar(v)
    try:
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return str(v)
            # 4 sig-figs, strip trailing zeros
            s = f"{v:.4g}"
            return s
        s = str(v)
        if len(s) > 64:
            s = s[:61] + "..."
        return s
    except Exception:
        return str(v)


def log_step_tsv(payload: Dict[str, Any], *, file=None) -> None:
    """Print a compact TSV line for a log payload (includes step if present)."""
    if file is None:
        file = sys.stdout
    parts: List[str] = []
    step = payload.get("step")
    if step is not None:
        parts.extend(["step", _fmt_val(step)])
    # Keep a stable order: common keys first, then the rest alphabetically
    preferred = ["loss", "lr"]
    keys = [k for k in preferred if k in payload and k != "step"]
    keys += sorted(k for k in payload.keys() if k not in set(keys + ["step"]))
    for k in keys:
        v = payload.get(k)
        parts.extend([k, _fmt_val(v)])
    print("\t".join(parts), file=file, flush=True)


class StdoutLogger:
    """Minimal stdout logger that plugs into hooks.

    Usage:
      hooks = make_stdout_hooks()
    """

    def __init__(self, file=None):
        self.file = file or sys.stdout

    # Hook signatures: (logs, ctx) / (metrics, ctx)
    def on_log(self, logs: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        log_step_tsv(logs, file=self.file)

    def on_eval_end(self, metrics: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        payload = {"step": ctx.get("step", None), **metrics}
        log_step_tsv(payload, file=self.file)

    def on_train_start(self, ctx: Dict[str, Any]) -> None:
        # Print a light banner with device and world info
        dev = ctx.get("device")
        rank = ctx.get("rank")
        world = ctx.get("world_size")
        cfg = ctx.get("config_json")
        print(f"[koochak] device={dev} rank={rank}/{world}", file=self.file, flush=True)
        if cfg is not None:
            try:
                import json

                print("[config] " + json.dumps(cfg, sort_keys=True), file=self.file, flush=True)
            except Exception:
                pass

    def on_train_end(self, ctx: Dict[str, Any]) -> None:
        print("[koochak] training finished", file=self.file, flush=True)


def make_stdout_hooks(file=None) -> Dict[str, List]:
    """Return a hooks dict wiring StdoutLogger into on_* events."""
    l = StdoutLogger(file=file)
    return {
        "on_train_start": [rank0_only(l.on_train_start)],
        "on_log": [rank0_only(l.on_log)],
        "on_eval_end": [rank0_only(l.on_eval_end)],
        "on_train_end": [rank0_only(l.on_train_end)],
    }
