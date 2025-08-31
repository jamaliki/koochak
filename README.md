# Koochak

A tiny, hackable, function‑first training loop for PyTorch — built to be easy to read, fork, and extend. It favors explicit functions and small modules over opaque classes or global state.

## Goals

- Functional core: a single `training_loop(...)` with a clear, compact signature.
- Hackable: pure-PyTorch, minimal magic, everything explicit via a small config mapping.
- Iterable-first: data is any iterable (finite or infinite). No hidden epoch semantics.
- Modern essentials: AMP, grad accumulation, grad clipping, logging, checkpointing, eval hooks, and DDP — each in small swappable modules.
- Low dependency: standard library + PyTorch (optional: torchvision for examples, tqdm for niceties, wandb for logging, pyyaml for configs).

## Repository Layout

- `koochak/`
  - `loop.py` – the core `training_loop` implementation (imports tiny helpers; loop remains minimal).
  - `core/`
    - `hooks.py` – tiny hook system: `merge/add/emit` and `rank0_only` wrapper.
    - `precision.py` – `autocast_context(mode, device)` and `Scaler(mode)`.
    - `dist.py` – DDP helpers: `init_process_group`, `barrier`, `rank/world_size`, `rank0`.
  - `data/`
    - `iterable.py` – `to_device(batch, device)`, `cycle(iterable)`, and `take(iterable, n)`.
    - `sharding.py` – `shard_iterable(iterable, rank, world_size)`.
  - `logging/`
    - `stdout.py` – compact TSV stdout logger + `make_stdout_hooks()`.
    - `csv.py` – `CSVLogger` and `make_csv_hooks(path)`.
    - `jsonl.py` – `JSONLLogger` and `make_jsonl_hooks(path)`.
    - `wandb_logger.py` – optional W&B hooks (lazy import), artifact upload.
  - `optim/`
    - `build.py` – tiny builders for optimizers/schedulers (supports cosine, step, plateau, cosine_warmup).
  - `storage/`
    - `checkpoint.py` – checkpoint save/load (atomic), `latest`, `best`.
    - `atomic.py` – atomic file writer.
    - `fs.py` – small FS utilities (`mkdir_p`, `latest`, `best`).
    - `pruning.py` – `prune_keep_last_k(dir, pattern, k)`.
  - `utils/`
    - `config.py` – `get(cfg, key, default)` and `as_dict(cfg)` helpers.
    - `device.py` – `get_device(cfg)` and `get_lr(optimizer)`.
    - `seed.py` – `set_all_seeds(seed)`, `make_worker_init_fn(seed)`, `get_rng_state()`.
    - `stats.py` – `SmoothedMeter`, `Throughput`, `EMA`.
    - `timeit.py` – `Timer` and `time_block(...)` context utilities.
- `examples/mnist/`
  - `config.yaml` – YAML-driven config split into `train`, `data`, `optim`, `wandb` sections.
  - `main.py` – minimal end-to-end example (YAML-only CLI), stdout hooks by default, optional W&B.
- `AGENTS.md` – running design notes + TODOs for contributors.

## Installation

- Python 3.9+
- PyTorch (CUDA optional): https://pytorch.org
- Optional: `pip install torchvision pyyaml wandb tqdm`

This repo is intentionally lightweight — it is not a packaged PyPI install. Import modules via the repo root (e.g., `python -m examples.mnist.main`).

## Quickstart (MNIST)

1) Configure YAML (defaults provided):

`examples/mnist/config.yaml`

- `train`: loop behavior (max_steps, log/eval/ckpt cadence, grad_accum, amp, seed, device, out_dir, keep_last_k, ddp flag).
- `data`: `data_dir`, `batch_size`, `num_workers`.
- `optim`: optimizer + scheduler (e.g., AdamW + cosine_warmup).
- `wandb`: set `enabled: true` to turn on W&B logging.

2) Run:

`python -m examples.mnist.main --config examples/mnist/config.yaml`

This will download MNIST (via torchvision), print TSV logs to stdout, periodically evaluate, and write atomic checkpoints to `train.out_dir` (e.g., `./runs/mnist/step000000000.pt` and `latest.pt`).


## Configuration

By default, Koochak enforces strict configuration to minimize surprises.

- Strict mode: unknown YAML keys cause an immediate error before training; unused `train.*` keys cause an error after training. To relax, set `train.strict_config: false`.
- Warnings: if strict is disabled, unknown/unused keys print rank-0 warnings when `train.config_warn_unknown: true` (default true).

Example YAML toggles:

```
train:
  strict_config: true       # default
  config_warn_unknown: true # default, applies when strict_config=false
```

At startup, a brief config summary prints sections present, unknown keys (if any), and strict status.



## Generic CLI

Koochak ships a generic YAML-driven CLI so you can run training without custom scripts.

Run:

`python -m koochak.cli.train --config path/to/your_config.yaml`

Your YAML must include an `entry` section to locate user code, plus the standard sections:

```
entry:
  model: your_pkg.model_defs:make_model     # returns nn.Module
  dataset: your_pkg.data:train_dataset      # returns iterable (or DataLoader)
  step: your_pkg.train:step_fn              # def step_fn(model, batch, ctx) -> dict
  eval_dataset: your_pkg.data:val_dataset   # optional
  eval_fn: your_pkg.train:eval_fn           # optional

train: { ... }
data:  { ... }
optim: { optimizer: {...}, scheduler: {...} }
logging: { csv_path: ..., jsonl_path: ... }
wandb: { enabled: false, project: ... }
```

The CLI applies the same strict config validation and pre-run summary, builds the optimizer/scheduler from `optim`, attaches stdout/CSV/JSONL/W&B hooks, resumes from the latest checkpoint under `train.out_dir`, and calls `training_loop`.


## Core API

`koochak/loop.py` exposes:

```
training_loop(
  *,
  model: nn.Module,
  dataset: Iterable,                 # any iterable (finite or infinite)
  step_fn: Callable,                 # returns {"loss": Tensor, ...}
  optimizer: Optimizer,
  scheduler: Optional[_LRScheduler] = None,
  config: Mapping[str, Any],
  checkpoint_dict: Optional[Dict[str, Any]] = None,
  eval_dataset: Optional[Iterable] = None,
  eval_fn: Optional[Callable] = None,
  hooks: Optional[Dict[str, list[Callable]]] = None,
) -> Dict[str, Any]
```

- `step_fn(model, batch, ctx)` returns `{"loss": Tensor, ...}`; any additional scalar values are logged.
- `ctx` contains `device`, `rank/world_size`, `autocast`, `scaler`, and `config_json`.
- The loop handles gradient accumulation, AMP, optional grad clipping, scheduler stepping (per `config.scheduler_step`), evaluation hooks, and checkpointing.
- Returns a plain checkpoint dict sufficient to resume.

Minimal step_fn example:

```
def step_fn(model, batch, ctx):
    x, y = batch["x"], batch["y"]
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    acc = (logits.argmax(-1) == y).float().mean()
    return {"loss": loss, "acc": acc}
```

## Hooks and Logging

- Create hooks by event name: `{"on_log": [fn], "on_eval_end": [fn]}`.
- Built-in hooks:
  - `koochak.logging.stdout.make_stdout_hooks()` – TSV prints; rank-0 only.
  - `koochak.logging.csv.make_csv_hooks(path)` – append metrics to CSV; rank-0 only.
  - `koochak.logging.jsonl.make_jsonl_hooks(path)` – one JSON per line; rank-0 only.
  - `koochak.logging.wandb_logger.make_wandb_hooks(cfg)` – W&B logging/artifacts; rank-0 only.
- Compose hooks with `koochak.core.hooks.merge(a, b)`. Gate any custom hook via `koochak.core.hooks.rank0_only(fn)` to ensure single-emission under DDP.

YAML-driven logging (example):

```
logging:
  csv_path: ./runs/mnist/log.csv
  jsonl_path: ./runs/mnist/log.jsonl
wandb:
  enabled: false
```

If `csv_path`/`jsonl_path` are omitted, the MNIST example defaults to `<train.out_dir>/log.csv` and `<train.out_dir>/log.jsonl`.

## Checkpointing

- `koochak.storage.checkpoint.save(ckpt, path, keep_last_k)` performs atomic writes, keeps only the last `k` step-checkpoints, and maintains `latest.pt`.
- `koochak.storage.checkpoint.load(path)` loads to CPU.
- `koochak.storage.checkpoint.latest(dir)` returns `latest.pt` if present or the most recent step checkpoint.
- `koochak.storage.checkpoint.best(dir, key)` selects the lowest metric across checkpoints.

DDP compatibility:
- The loop saves the underlying module weights when the model is wrapped in `DistributedDataParallel` (i.e., uses `model.module.state_dict()`), making checkpoints portable across single-GPU and DDP.
- When loading manually, use the provided helpers if your loading target differs in wrapping:
  - `from koochak.storage.checkpoint import match_state_dict_to_model`
  - `target = getattr(model, 'module', model)`
  - `target.load_state_dict(match_state_dict_to_model(target, ckpt['model']))`

## Resuming Across DDP/Single GPU

When moving between single-GPU and DDP runs, key prefixes can differ (`module.`). The loop saves the underlying module weights for portability, but if you’re loading manually, use the helpers:

````
from koochak.storage.checkpoint import load, match_state_dict_to_model

ckpt = load(path)
target = getattr(model, 'module', model)
state = match_state_dict_to_model(target, ckpt['model'])
target.load_state_dict(state)
````


Checkpoint dict fields include: `step`, `model`, `optimizer`, `scheduler` (optional), `scaler` (optional), `config`, RNG state, `wall_time`, and `metrics`.

## Distributed (DDP)

- Initialize distributed (outside the loop):

```
from koochak.core import dist as dist_lib
if need_ddp:
    dist_lib.init_process_group(backend="nccl")   # or "gloo" on CPU
```

- In your YAML, set `train.ddp: true` to shard the iterable by rank/world via `koochak.data.sharding.shard_iterable`.
- Only rank 0 writes checkpoints and logs via built-in hooks; barriers enclose checkpoint steps to align rank progress.
- Launch with torchrun as usual:

`torchrun --nproc_per_node=8 -m examples.mnist.ddp_main --config examples/mnist/config.yaml`

The DDP launcher:
- Calls `init_process_group(backend=...)` and sets the current CUDA device from `LOCAL_RANK`.
- Forces `train.ddp: true` and keeps other training settings from YAML.
- Uses the same logging configuration (stdout/CSV/JSONL and optional W&B).

## Optimizers and Schedulers

- `koochak.optim.build.build_optimizer(params, cfg)` supports `adamw`, `adam`, `sgd`.
- `koochak.optim.build.build_scheduler(optimizer, cfg, train_cfg)` supports `cosine`, `step`, `plateau`, and `cosine_warmup`.

Example YAML (snippets):

```
optim:
  optimizer:
    name: AdamW
    lr: 0.0003
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 100
    T_max: null   # falls back to train.max_steps
    eta_min: 0.0
```

## Reproducibility

- `koochak.utils.seed.set_all_seeds(seed)` sets Python/NumPy/Torch seeds and (optionally) CUDA seeds.
- `koochak.utils.seed.make_worker_init_fn(seed)` seeds DataLoader workers deterministically.
- RNG state is stored in checkpoints (`get_rng_state()`), so resumed runs continue deterministically.

## Contributing

- Start with `README.md` and skim `design_doc.md` to understand the philosophy.
- See `AGENTS.md` for current implementation notes and a living TODO list. Keep it up to date as you work.
- Code style: clear, minimal, single-purpose modules. Favor functions and plain dicts over classes.

## Tests

Unit tests live under `tests/`. Use your preferred runner (e.g., `pytest`) from the repo root:

```
pip install pytest
pytest -q
```

## License

MIT (see `LICENSE`).
