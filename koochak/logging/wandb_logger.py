from __future__ import annotations

import re, hashlib
from typing import Any, Dict, List
from ..core.hooks import rank0_only
from ..storage.naming import make_checkpoint_aliases


def _run_id_from_name(name: str) -> str:
    """
    Deterministically map a human run name -> a stable W&B run id.
    Returns a lowercase hex string (SHA1), which is W&B-safe.
    """
    if not name or not isinstance(name, str):
        raise RuntimeError("W&B resume-by-name requires cfg.name to be a non-empty string.")
    # Normalize the name to avoid accidental changes (spaces, case, punctuation)
    canonical = re.sub(r"[^A-Za-z0-9-_]+", "-", name.strip().lower())
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()  # 40 chars


def make_wandb_hooks(cfg) -> Dict[str, List]:
    """Factory returning W&B hook callbacks per design_doc.

    Soft-depends on `wandb`; returns an empty dict if unavailable or disabled.
    """
    import wandb  # type: ignore

    # If a simple dict-like is passed, mimic attribute access
    class _Cfg:
        def __init__(self, c):
            self._c = c

        def __getattr__(self, name):
            if isinstance(self._c, dict):
                return self._c.get(name)
            return getattr(self._c, name)

    cfg = _Cfg(cfg)

    # Track best metrics to alias artifacts accordingly
    best_values: Dict[str, float] = {}
    best_steps: Dict[str, int] = {}

    def _cfg_get(obj, key: str, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
        except Exception:
            pass
        return getattr(obj, key, default)

    def on_train_start(ctx: Dict[str, Any]):
        if getattr(cfg, "enabled", True) is False:
            return
        settings = wandb.Settings(start_method="thread")

        run_name = getattr(cfg, "name", None)
        run_id = _run_id_from_name(run_name)

        wandb.init(
            project=getattr(cfg, "project", "koochak"),
            entity=getattr(cfg, "entity", None),
            name=run_name,                                
            group=getattr(cfg, "group", None),
            job_type=getattr(cfg, "job_type", None),
            tags=getattr(cfg, "tags", None),
            notes=getattr(cfg, "notes", None),
            mode=getattr(cfg, "mode", "online"),
            dir=getattr(cfg, "dir", None),
            id=run_id,                                    
            resume="allow",                               
            settings=settings,
            config=ctx.get("config_json") or ctx.get("config"),
            allow_val_change=True,                        
        )

        if getattr(cfg, "watch_model", True):
            model = ctx.get("model")
            if model is not None:
                if getattr(cfg, "watch_unwrap_ddp", True) and hasattr(model, "module"):
                    try:
                        model = model.module
                    except Exception:
                        pass

                watch_kwargs = {
                    "log": getattr(cfg, "watch_log", "all") or "all",
                    "log_freq": int(getattr(cfg, "watch_log_freq", 100) or 100),
                    "log_graph": bool(getattr(cfg, "watch_log_graph", False)),
                }

                try:
                    watch_fn = getattr(wandb, "watch_model", None)
                    if callable(watch_fn):
                        watch_fn(model, criterion=None, **watch_kwargs)
                    else:
                        wandb.watch(model, **watch_kwargs)
                except Exception as exc:
                    msg = (
                        "wandb.watch/watch_model failed; gradients/parameters won't be tracked. "
                        f"Error: {exc}"
                    )
                    try:
                        wandb.termwarn(msg)
                    except Exception:
                        print(f"[wandb] {msg}")

    def on_log(logs: Dict[str, Any], ctx: Dict[str, Any]):
        # Avoid W&B "out of order step" warnings when eval logs at same step.
        # If this step will also emit eval metrics, delay the commit until on_eval_end.
        step = int(ctx.get("step", 0))
        train_cfg = ctx.get("train_cfg") or {}
        try:
            eval_every = int(_cfg_get(train_cfg, "eval_every", 0))
        except Exception:
            eval_every = 0
        will_eval = eval_every > 0 and (step % eval_every == 0)
        wandb.log(logs, step=step, commit=not will_eval)

    def on_eval_end(metrics: Dict[str, Any], ctx: Dict[str, Any]):
        # Commit at eval time so training+eval logs share the same step without warnings
        wandb.log(metrics, step=ctx.get("step"), commit=True)
        run = wandb.run
        if run is None:
            return
        for k, v in metrics.items():
            prev = run.summary.get(f"best/{k}", float("inf"))
            try:
                prev = float(prev)
                val = float(v)
            except Exception:
                continue
            run.summary[f"best/{k}"] = min(prev, val)
            # Track best step for aliasing artifacts later
            if val <= prev:
                best_values[k] = val
                step = int(ctx.get("step", -1))
                if step >= 0:
                    best_steps[k] = step

    def _artifact_base_name(run) -> str:
        prefix = getattr(cfg, "artifact_name_prefix", None) or getattr(cfg, "artifact_name", None) or "model"
        # Prefer stable run id for versioning within a collection
        rid = getattr(run, "id", None) or getattr(run, "name", None) or "unknown"
        return f"{prefix}-{rid}"

    def on_checkpoint(path: str, ckpt: Dict[str, Any], ctx: Dict[str, Any]):
        if not getattr(cfg, "log_artifacts", True):
            return
        run = wandb.run
        if run is None:
            return
        art_type = getattr(cfg, "artifact_type", "model")
        name = _artifact_base_name(run)
        art = wandb.Artifact(name=name, type=art_type, metadata={
            "step": ctx.get("step"),
        })
        art.add_file(path)
        # Determine if this step is best for any tracked key
        step = int(ctx.get("step", -1))
        best_keys_for_step: List[str] = [k for k, s in best_steps.items() if s == step]
        aliases = make_checkpoint_aliases(step if step >= 0 else None, include_latest=True, best_keys=best_keys_for_step or None)
        wandb.log_artifact(art, aliases=aliases)

    def on_train_end(ctx: Dict[str, Any]):
        try:
            wandb.finish()
        except Exception:
            pass

    return {
        "on_train_start": [rank0_only(on_train_start)],
        "on_log": [rank0_only(on_log)],
        "on_eval_end": [rank0_only(on_eval_end)],
        "on_checkpoint": [rank0_only(on_checkpoint)],
        "on_train_end": [rank0_only(on_train_end)],
    }
