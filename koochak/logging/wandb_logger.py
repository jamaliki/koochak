from __future__ import annotations

from typing import Any, Dict, List


def make_wandb_hooks(cfg) -> Dict[str, List]:
    """Factory returning W&B hook callbacks per design_doc.

    Soft-depends on `wandb`; returns an empty dict if unavailable or disabled.
    """
    try:
        import wandb  # type: ignore
    except Exception:
        return {}

    # If a simple dict-like is passed, mimic attribute access
    class _Cfg:
        def __init__(self, c):
            self._c = c

        def __getattr__(self, name):
            if isinstance(self._c, dict):
                return self._c.get(name)
            return getattr(self._c, name)

    cfg = _Cfg(cfg)

    def on_train_start(ctx: Dict[str, Any]):
        if getattr(cfg, "enabled", True) is False:
            return
        settings = wandb.Settings(start_method="thread")
        wandb.init(
            project=getattr(cfg, "project", "koochak"),
            entity=getattr(cfg, "entity", None),
            name=getattr(cfg, "name", None),
            group=getattr(cfg, "group", None),
            job_type=getattr(cfg, "job_type", None),
            tags=getattr(cfg, "tags", None),
            notes=getattr(cfg, "notes", None),
            mode=getattr(cfg, "mode", "online"),
            dir=getattr(cfg, "dir", None),
            resume=(getattr(cfg, "resume", "never") if getattr(cfg, "id", None) else "never"),
            id=getattr(cfg, "id", None),
            settings=settings,
            config=ctx.get("config_json"),
        )

    def on_log(logs: Dict[str, Any], ctx: Dict[str, Any]):
        wandb.log(logs, step=ctx.get("step"), commit=True)

    def on_eval_end(metrics: Dict[str, Any], ctx: Dict[str, Any]):
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

    def on_checkpoint(path: str, ckpt: Dict[str, Any], ctx: Dict[str, Any]):
        if not getattr(cfg, "log_artifacts", True):
            return
        art = wandb.Artifact("checkpoint", type="model")
        art.add_file(path)
        wandb.log_artifact(art, aliases=["latest", f"step-{ctx.get('step')}"])

    def on_train_end(ctx: Dict[str, Any]):
        try:
            wandb.finish()
        except Exception:
            pass

    return {
        "on_train_start": [on_train_start],
        "on_log": [on_log],
        "on_eval_end": [on_eval_end],
        "on_checkpoint": [on_checkpoint],
        "on_train_end": [on_train_end],
    }

