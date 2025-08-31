# Koochak

*A tiny, hackable, function‑first training loop for PyTorch.*

---

## 1) Philosophy & Goals

**Why**: Big trainer classes (Lightning, etc.) can feel opaque when you just want to tinker. Koochak aims to be a **small, pragmatic toolkit** of **functions** that compose cleanly and are easy to fork. No magic, minimal global state, everything explicit.

**Primary goals**

* **Functional core**: a single `training_loop(...)` with a clear, compact signature.
* **Hackable**: pure-PyTorch, all knobs exposed via a simple `config` mapping. No hidden state machines.
* **Iterable-first**: datasets are **always** `IterableDataset` or any iterable; no epoch semantics assumed.
* **Modern essentials**: mixed precision, gradient accumulation, gradient clipping, logging, checkpointing, evaluation hooks, and DDP multi-GPU — all optional, each implemented in **<100 LOC** modules.
* **Low dependency**: standard library + PyTorch only (optional: `tqdm` for pretty bars).

**Non-goals**

* Replace all trainer frameworks.
* Provide a one-size-fits-all logger/visualizer. We ship tiny CSV/JSONL loggers you can swap out.
* Manage experiment registries or cloud storage.

---

## 2) Package layout

```
koochak/
  __init__.py
  core/
    loop.py              # training_loop (core), evaluate, hooks dispatch
    hooks.py             # event registry + emit/merge utils
    precision.py         # autocast + (no-op) scaler helpers
    dist.py              # DDP helpers (init, rank/world, barrier)
  data/
    iterable.py          # cycle(), take(), to_device(), collate helpers
    sharding.py          # shard_iterable(iterable, rank, world_size)
    workers.py           # worker seeding for DataLoader
  optim/
    build.py             # tiny builders for optimizer/scheduler (optional)
    schedulers.py        # cosine, warmup, plateau wrappers
  logging/
    stdout.py            # StdoutLogger + formatting
    csv.py               # CSVLogger (optional)
    jsonl.py             # JSONLLogger (optional)
    wandb_logger.py      # W&B hooks (optional dependency, lazy import)
  storage/               # (avoid naming this `io` to not shadow stdlib)
    checkpoint.py        # save/load checkpoint dicts
    atomic.py            # atomic_write(path, tmp_suffix)
    pruning.py           # prune_keep_last_k(dir, pattern, k)
    fs.py                # small FS utilities (mkdir_p, latest, best)
  utils/
    stats.py             # SmoothedMeter, Throughput, EMA
    timeit.py            # scoped timers
    seed.py              # python/np/torch seeding
    device.py            # rank0(), get_device(), get_lr()
  examples/
    minimal_mlp.py       # 40-line end-to-end demo
    ddp_iterable.py      # multi-GPU IterableDataset demo
  cli/
    __init__.py
    train.py             # thin CLI wrapper around core.loop.training_loop
```

> Everything remains **function-first**, just organized into **small sub-packages** so it’s easy to find and swap bits. We avoid a package named `io` to prevent collisions with Python’s stdlib `io` module.

---

## 3) Configuration (explicit and boring)

A plain dict or small dataclasses; nothing magical. All defaults live in code.

```python
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, List

@dataclass
class WandBConfig:
    enabled: bool = True
    project: str = "koochak"
    entity: Optional[str] = None
    name: Optional[str] = None      # run name
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: Literal["online", "offline"] = "online"  # use WANDB_MODE=offline as well
    dir: Optional[str] = None       # where to store wandb files (defaults to cwd)
    resume: Literal["never", "allow", "must"] = "allow"
    id: Optional[str] = None        # stable run id for resuming
    log_artifacts: bool = True      # upload checkpoints as artifacts

@dataclass
class TrainConfig:
    # Loop limits
    max_steps: int = 100_000
    log_every: int = 100
    eval_every: int = 5_000
    ckpt_every: int = 5_000

    # Optim / schedule
    grad_accum: int = 1
    grad_clip_norm: Optional[float] = None
    scheduler_step: Literal["step", "eval", "never"] = "step"

    # Precision
    amp: Literal["fp32", "fp16", "bf16"] = "fp32"

    # Distributed
    ddp: bool = False
    ddp_backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    find_unused_parameters: bool = False

    # Misc
    seed: int = 42
    compile: bool = False  # torch.compile for model
    device: str = "cuda"   # or "cpu"
    out_dir: str = "./runs/exp0"
    keep_last_k: int = 3   # checkpoints to retain

    # Logging
    wandb: Optional[WandBConfig] = None  # enable W&B if not None
    stdout_logging: bool = True          # keep simple stdout logs
```

Pass either a dict or these dataclasses. The loop does not read env vars except when W\&B is enabled (respecting `WANDB_*`).

---

## 4) Core APIs

### 4.1 `training_loop` (the heart)

```python
from typing import Iterable, Mapping, Callable, Optional, Any, Dict
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

StepFn = Callable[[nn.Module, Any, Mapping[str, Any]], Dict[str, Any]]
EvalFn = Callable[[nn.Module, Iterable, Mapping[str, Any]], Dict[str, float]]

def training_loop(
    *,
    model: nn.Module,
    dataset: Iterable,                # MUST be an iterable (finite or infinite)
    step_fn: StepFn,                  # returns {"loss": Tensor, ...logs}
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    config: Mapping[str, Any],
    checkpoint_dict: Optional[Dict[str, Any]] = None,  # for resume
    eval_dataset: Optional[Iterable] = None,
    eval_fn: Optional[EvalFn] = None,
    hooks: Optional[Dict[str, list[Callable]]] = None,  # event -> [callbacks]
) -> Dict[str, Any]:
    """Runs until `config.max_steps` or dataset stops. Returns final ckpt dict."""
```

**Contract**

* `step_fn(model, batch, ctx)` **must** return a dict containing at least `{"loss": Tensor}`. It can add arbitrary scalars to log.
* `ctx` is a small read-only mapping with useful handles (rank, step, device, autocast context, scaler, etc.).
* `dataset` is any iterable of batches. If finite, you can wrap with `data.cycle(dataset)`.
* Loop maintains and returns a plain `checkpoint_dict` with everything needed to resume.

### 4.2 Step function (you own your forward pass)

```python
def step_fn(model, batch, ctx):
    # model in train mode, autocast already active
    logits = model(batch["x"])  
    loss = ctx["criterion"](logits, batch["y"])  # you decide the loss
    return {"loss": loss, "acc": (logits.argmax(-1) == batch["y"]).float().mean()}
```

### 4.3 Evaluation

```python
def eval_fn(model, eval_iterable, ctx):
    model.eval()
    n, total = 0, 0.0
    with torch.no_grad(), ctx["autocast"]:
        for batch in eval_iterable:
            out = model(batch["x"])
            total += ctx["criterion_eval"](out, batch["y"]).item()
            n += 1
    return {"val_loss": total / max(n, 1)}
```

### 4.4 Hooks (tiny event system)

Hooks are just lists of callables keyed by event name:

* `on_train_start(ctx)`
* `on_step_end(logs, ctx)`
* `on_log(logs, ctx)`
* `on_eval_end(metrics, ctx)`
* `on_checkpoint(ckpt_path, ckpt_dict, ctx)`
* `on_train_end(ctx)`
* `on_exception(exc, ctx)`

They receive **data only**, never mutate loop state except via the filesystem or their own closures.

---

## 5) Distributed & precision

**Distributed (DDP)**

* `dist.init_process_group(...)` helper; only rank 0 writes to disk/stdout.
* Wrap model with `torch.nn.parallel.DistributedDataParallel` when `config.ddp=True`.
* Gradient accumulation uses `no_sync` on non-final micro-steps.
* Iterable datasets are sharded by `data.shard_iterable(it, rank, world_size)` which yields every `world_size`-th element starting at `rank`.
* `barrier()` before/after checkpointing for clean resumes.

**Precision**

* `precision.autocast_context(cfg.amp)` yields `nullcontext` (fp32), `autocast(dtype=torch.float16)`, or `autocast(dtype=torch.bfloat16)`.
* `precision.Scaler(cfg.amp)` returns a `GradScaler` for fp16, or a dummy no-op scaler for bf16/fp32.

---

## 6) Iterable data utilities

```python
# Make a finite iterable infinite by cycling
cycle(iterable)

# Deterministic sharding for IterableDataset (DDP-friendly)
shard_iterable(iterable, rank: int, world_size: int)

# Worker seeding helper for DataLoader(num_workers>0)
make_worker_init_fn(base_seed: int)
```

Design notes:

* No `DistributedSampler` (doesn’t support IterableDataset semantics). We use simple striding.
* Users who already implement internal sharding can skip `shard_iterable`.

---

## 7) Checkpointing (plain dicts)

**Layout**

```python
ckpt = {
  "step": int,
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "scheduler": scheduler.state_dict() if scheduler else None,
  "scaler": scaler.state_dict() if scaler else None,
  "config": asdict(config),
  "rng": {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.get_rng_state(),
    "torch_cuda": torch.cuda.get_rng_state_all(),
  },
  "wall_time": float,  # seconds since epoch
  "metrics": {"best": float, ...},
}
```

**API**

* `checkpoint.save(ckpt, path, keep_last_k=3)` — atomic write (tmp file + rename), prunes old checkpoints, rank0 only.
* `checkpoint.load(path)` — returns dict, caller loads states.
* Optional: `checkpoint.latest(dir)` and `checkpoint.best(dir, key="val_loss")`.

Resume is trivial: pass the loaded dict to `training_loop(..., checkpoint_dict=ckpt)`.

---

## 8) Logging & stats

We keep **stdout logging + stats**, and add **Weights & Biases** (optional, lazy import). In DDP, **only rank 0** logs.

### Built-ins

* `utils.stats.SmoothedMeter(window=100)` — rolling mean/std/min/max.
* `utils.stats.Throughput()` — items/sec using batch size and wall time.
* `logging.stdout.log_step(logs)` — prints compact TSV.
* `logging.csv.CSVLogger` / `logging.jsonl.JSONLLogger` — append-only files.

### W\&B (optional)

Enable by setting `cfg.wandb = WandBConfig(...)`. We expose a **hook factory** that returns event callbacks, no classes.

```python
# logging/wandb_logger.py
from typing import Dict, Any

def make_wandb_hooks(cfg) -> Dict[str, list]:
    try:
        import wandb
    except ImportError:  # soft-dependency
        return {}

    def on_train_start(ctx):
        if not cfg.enabled:
            return
        settings = wandb.Settings(start_method="thread")
        wandb.init(
            project=cfg.project, entity=cfg.entity, name=cfg.name,
            group=cfg.group, job_type=cfg.job_type, tags=cfg.tags,
            notes=cfg.notes, mode=cfg.mode, dir=cfg.dir,
            resume=cfg.resume if cfg.id else "never", id=cfg.id,
            settings=settings,
            config=ctx["config_json"],  # full run config for reproducibility
        )

    def on_log(logs, ctx):
        # W&B expects plain scalars; you can pre-filter here if needed
        wandb.log(logs, step=ctx["step"], commit=True)

    def on_eval_end(metrics, ctx):
        wandb.log(metrics, step=ctx["step"], commit=True)
        for k, v in metrics.items():
            wandb.run.summary[f"best/{k}"] = min(wandb.run.summary.get(f"best/{k}", float("inf")), v)

    def on_checkpoint(path, ckpt, ctx):
        if not cfg.log_artifacts:
            return
        art = wandb.Artifact("checkpoint", type="model")
        art.add_file(path)
        wandb.log_artifact(art, aliases=["latest", f"step-{ctx['step']}"])

    def on_train_end(ctx):
        wandb.finish()

    return {
        "on_train_start": [on_train_start],
        "on_log": [on_log],
        "on_eval_end": [on_eval_end],
        "on_checkpoint": [on_checkpoint],
        "on_train_end": [on_train_end],
    }
```

**Using the hooks**

```python
from koochak.core import hooks as hooks_lib
from koochak.logging.wandb_logger import make_wandb_hooks

hooks = hooks_lib.merge(hooks, make_wandb_hooks(cfg.wandb) if cfg.wandb else {})
```

**Notes**

* Supports offline mode via `cfg.wandb.mode = "offline"` or `WANDB_MODE=offline`.
* W\&B logging is minimal: we just ship raw dicts from `on_log`/`on_eval_end`.
* For performance, keep `log_every` sane; W\&B is on the critical path only at those steps.

---

## 9) Step lifecycle (pseudocode)

```python
# inside training_loop
ctx = build_ctx(model, optimizer, scheduler, config, scaler, device, rank, world_size)

it = iter(dataset)  # MUST be iterable
if ddp: it = data.shard_iterable(it, rank, world_size)
if finite: it = cycle(it)  # user decides; we provide helper

for step in range(resume_step, config.max_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Gradient accumulation
    total_loss = 0.0
    for micro in range(config.grad_accum):
        with autocast:
            batch = next(it)
            out = step_fn(model, to_device(batch, device), ctx)
            loss = out["loss"] / config.grad_accum
        scaler.scale(loss).backward()
        total_loss += float(loss.detach())

    if config.grad_clip_norm:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), config.grad_clip_norm)

    scaler.step(optimizer)
    scaler.update()

    if scheduler and config.scheduler_step == "step":
        scheduler.step()

    # Logging / hooks
    logs = {"loss": total_loss, "lr": get_lr(optimizer), **out, "step": step}
    if rank0 and step % config.log_every == 0:
        emit("on_log", logs, ctx)
    emit("on_step_end", logs, ctx)

    # Periodic eval
    if eval_fn and eval_dataset and step % config.eval_every == 0:
        metrics = eval_fn(model, eval_dataset, ctx)
        if scheduler and config.scheduler_step == "eval":
            scheduler.step(metrics.get("val_loss", None))
        if rank0: emit("on_eval_end", metrics, ctx)

    # Periodic checkpoint
    if rank0 and step % config.ckpt_every == 0:
        ckpt = make_ckpt_dict(...)
        path = checkpoint.save(ckpt, f"{config.out_dir}/step{step:09d}.pt",
                               keep_last_k=config.keep_last_k)
        emit("on_checkpoint", path, ckpt, ctx)
```

---

## 10) Error handling & determinism

* `try/except` around the main loop calls `on_exception(exc, ctx)` and re-raises by default.
* Seeding: set Python, NumPy, and Torch seeds; seed DataLoader workers deterministically (helper provided).
* Optional `torch.autograd.set_detect_anomaly(True)` in config for debugging.

---

## 11) Example (single GPU, minimal)

```python
from koochak.loop import training_loop
from koochak.config import TrainConfig
from koochak.checkpoint import load

model = MLP(...).to("cuda")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100_000)

def data_stream():
    while True:
        x, y = sample_batch()
        yield {"x": x.cuda(non_blocking=True), "y": y.cuda(non_blocking=True)}

cfg = TrainConfig(max_steps=20_000, amp="bf16", log_every=50, eval_every=5000)
ckpt = load("./runs/exp0/latest.pt") if os.path.exists(...) else None

training_loop(
    model=model,
    dataset=data_stream(),
    step_fn=step_fn,
    optimizer=opt,
    scheduler=sched,
    config=cfg,
    checkpoint_dict=ckpt,
)
```

---

## 12) Testing checklist

* Unit: precision helpers (fp16/bf16), no-op scaler, checkpoint save/load round-trip.
* Unit: shard\_iterable correctness across ranks (1..N) + short cycling.
* Unit: grad accumulation vs batch doubling equivalence on a toy model.
* Integration: resume from checkpoint mid-run (same loss curve within tolerance).
* Integration: DDP correctness (sum of losses divided by world\_size equals single-GPU baseline).

---

## 13) Roadmap (strictly optional)

* Optional FSDP wrapper (still function-first, config flag `fsdp=True`).
* Async checkpoint writer (background thread) with bounded queue.
* Built-in early stopping (small function using hooks).
* Tiny TensorBoard writer as an optional logger function.

---

## 14) Design invariants (hackability guardrails)

* **No** singletons or global registries.
* All side effects (I/O, printing) are gated by `rank0` checks.
* Hook APIs pass **plain dicts** and **immutables**; they never own control flow.
* The loop’s return value is **exactly** the checkpoint dict you’d need to resume.
* Every module aims to be **readable in one screen** of code.

---

**That’s it.** Small on purpose. Clone, fork, and play.

