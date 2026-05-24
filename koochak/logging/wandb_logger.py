from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping

from .. import config as config_lib
from ..core.hooks import rank0_only
from ..storage.naming import make_checkpoint_aliases


def _run_id_from_name(name: str) -> str:
    """
    Deterministically map a human run name -> a stable W&B run id.
    Returns a lowercase hex string (SHA1), which is W&B-safe.
    """
    if not name or not isinstance(name, str):
        raise RuntimeError("W&B stable-id-by-name requires cfg.name to be a non-empty string.")
    # Normalize the name to avoid accidental changes (spaces, case, punctuation)
    canonical = re.sub(r"[^A-Za-z0-9-_]+", "-", name.strip().lower())
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()  # 40 chars


def _compile_enabled_from_ctx(ctx: Mapping[str, Any]) -> bool:
    train_cfg = ctx.get("train_cfg") or {}
    compile_cfg = config_lib.get(train_cfg, "compile", None)
    if isinstance(compile_cfg, Mapping):
        return bool(compile_cfg.get("enabled", True))
    if isinstance(compile_cfg, bool):
        return bool(compile_cfg)
    return bool(compile_cfg)


def make_wandb_hooks(cfg) -> Dict[str, List]:
    """Factory returning W&B hook callbacks per design_doc.

    Soft-depends on `wandb`; returns an empty dict if unavailable or disabled.
    """
    import wandb  # type: ignore

    # Read-through accessor handling both Mapping and attr-bearing cfgs uniformly.
    def get(key: str, default: Any = None) -> Any:
        return config_lib.get(cfg, key, default)

    # Track best metrics to alias artifacts accordingly
    best_values: Dict[str, float] = {}
    best_steps: Dict[str, int] = {}

    def on_train_start(ctx: Dict[str, Any]):
        if get("enabled", True) is False:
            return
        settings = wandb.Settings()

        run_name = get("name", None)
        resume_mode = get("resume", None)
        run_id = get("id", None)
        if not run_id and run_name:
            run_id = _run_id_from_name(run_name)

        wandb.init(
            project=get("project", "koochak"),
            entity=get("entity", None),
            name=run_name,
            group=get("group", None),
            job_type=get("job_type", None),
            tags=get("tags", None),
            notes=get("notes", None),
            mode=get("mode", "online"),
            dir=get("dir", None),
            id=run_id,
            resume=resume_mode,
            settings=settings,
            config=ctx.get("config_json") or ctx.get("config"),
            allow_val_change=True,
        )

        watch_model = get("watch_model", None)
        if watch_model is None:
            watch_model = not _compile_enabled_from_ctx(ctx)

        if not watch_model:
            return
        model = ctx.get("model")
        if model is None:
            return
        if get("watch_unwrap_ddp", True):
            model = getattr(model, "module", model)
        watch_kwargs = {
            "log": get("watch_log", "all") or "all",
            "log_freq": int(get("watch_log_freq", 100) or 100),
            "log_graph": bool(get("watch_log_graph", False)),
        }
        try:
            watch_fn = getattr(wandb, "watch_model", None)
            if callable(watch_fn):
                watch_fn(model, criterion=None, **watch_kwargs)
            else:
                wandb.watch(model, **watch_kwargs)
        except (RuntimeError, TypeError, ValueError) as exc:
            msg = (
                "wandb.watch/watch_model failed; gradients/parameters won't be tracked. "
                f"Error: {exc}"
            )
            termwarn = getattr(wandb, "termwarn", None)
            if callable(termwarn):
                termwarn(msg)
            else:
                print(f"[wandb] {msg}")

    def on_log(logs: Dict[str, Any], ctx: Dict[str, Any]):
        # Avoid W&B "out of order step" warnings when eval logs at same step.
        # If this step will also emit eval metrics, delay the commit until on_eval_end.
        step = int(ctx.get("step", 0))
        train_cfg = ctx.get("train_cfg") or {}
        eval_every_raw = config_lib.get(train_cfg, "eval_every", 0)
        try:
            eval_every = int(eval_every_raw) if eval_every_raw is not None else 0
        except (TypeError, ValueError):
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
            prev_raw = run.summary.get(f"best/{k}", float("inf"))
            try:
                prev = float(prev_raw)
                val = float(v)
            except (TypeError, ValueError):
                continue
            run.summary[f"best/{k}"] = min(prev, val)
            if val <= prev:
                best_values[k] = val
                step = int(ctx.get("step", -1))
                if step >= 0:
                    best_steps[k] = step

    def _artifact_base_name(run) -> str:
        prefix = get("artifact_name_prefix", None) or get("artifact_name", None) or "model"
        rid = getattr(run, "id", None) or getattr(run, "name", None) or "unknown"
        return f"{prefix}-{rid}"

    def on_checkpoint(path: str, ckpt: Dict[str, Any], ctx: Dict[str, Any]):
        if not get("log_artifacts", True):
            return
        run = wandb.run
        if run is None:
            return
        art_type = get("artifact_type", "model")
        name = _artifact_base_name(run)
        art = wandb.Artifact(name=name, type=art_type, metadata={"step": ctx.get("step")})
        art.add_file(path)
        step = int(ctx.get("step", -1))
        best_keys_for_step: List[str] = [k for k, s in best_steps.items() if s == step]
        aliases = make_checkpoint_aliases(
            step if step >= 0 else None,
            include_latest=True,
            best_keys=best_keys_for_step or None,
        )
        wandb.log_artifact(art, aliases=aliases)

    def on_train_end(ctx: Dict[str, Any]):
        finish = getattr(wandb, "finish", None)
        if callable(finish):
            try:
                finish()
            except RuntimeError:
                # wandb may already be torn down (e.g., crashed sub-run); nothing to do.
                pass

    return {
        "on_train_start": [rank0_only(on_train_start)],
        "on_log": [rank0_only(on_log)],
        "on_eval_end": [rank0_only(on_eval_end)],
        "on_checkpoint": [rank0_only(on_checkpoint)],
        "on_train_end": [rank0_only(on_train_end)],
    }
