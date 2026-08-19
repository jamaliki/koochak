# Koochak

A tiny, hackable, function‑first training loop for PyTorch. Built to be easy to read, fork, and extend. It favors explicit functions and small modules over opaque classes or global state.

## Goals

- Functional core: a single `training_loop(...)` with a clear, compact signature.
- Hackable: pure-PyTorch, minimal magic, everything explicit via a small config mapping.
- Iterable-first: data is any iterable (finite or infinite). No hidden epoch semantics.
- Modern essentials: AMP, grad accumulation, grad clipping, logging, checkpointing, eval hooks, and DDP, each in small swappable modules.
- Low dependency: standard library + PyTorch (OmegaConf for configs; optional: torchvision for examples, tqdm for niceties, wandb for logging).

## Repository Layout

- `koochak/`
  - `loop.py` – the core `training_loop` implementation (imports tiny helpers; loop remains minimal).
  - `config.py` – OmegaConf + dataclass config loader, defaults, and summary helpers.
  - `core/`
    - `hooks.py` – tiny hook system: `merge/add/emit` and `rank0_only` wrapper.
    - `precision.py` – `autocast_context(mode, device)` and `Scaler(mode)`.
    - `dist.py` – DDP helpers: `init_process_group`, `barrier`, `rank/world_size`, `rank0`.
  - `data/`
    - `iterable.py` – `to_device(batch, device)`, `cycle(iterable)`, and `take(iterable, n)`.
    - `sharding.py` – `shard_dataset(..., mode=...)`, `shard_iterable_dataset`, `shard_map_dataset`.
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
    - `config.py` – thin compatibility wrappers around `koochak.config` (`get/as_dict`).
    - `device.py` – `get_device(cfg)` and `get_lr(optimizer)`.
    - `seed.py` – `set_all_seeds(seed)`, `make_worker_init_fn(seed)`, `get_rng_state()`.
    - `stats.py` – `SmoothedMeter`, `Throughput`, `EMA`.
    - `timeit.py` – `Timer` and `time_block(...)` context utilities.
- `examples/mnist/`
  - `config.yaml` – YAML-driven config split into `train`, `data`, `optim`, `logging`, `wandb` sections.
  - `main.py` – minimal end-to-end example (YAML-only CLI), stdout hooks by default, optional W&B.
- `examples/config_template.yaml` – canonical template with all supported keys.
- `AGENTS.md` – running design notes + TODOs for contributors.

## Installation

- Python 3.9+
- PyTorch (CUDA optional): https://pytorch.org
- Required: `pip install omegaconf`
- Optional: `pip install torchvision wandb tqdm`

This repo is intentionally lightweight: it is not a packaged PyPI install. Import modules via the repo root (e.g., `python -m examples.mnist.main`).

## Quickstart (MNIST)

1) Configure YAML (defaults provided):

`examples/mnist/config.yaml`

- `train`: loop behavior (max_steps, log/eval/ckpt cadence, grad_accum, amp, seed, device, out_dir, keep_last_k, ddp flag).
- `data`: `data_dir`, `batch_size`, `num_workers`.
- `optim`: optimizer + scheduler (e.g., AdamW + cosine_warmup).
- `logging`: `csv_path`, `jsonl_path`.
- `wandb`: set `enabled: true` to turn on W&B logging.

2) Run:

`python -m examples.mnist.main --config examples/mnist/config.yaml`

This will download MNIST (via torchvision), print TSV logs to stdout, periodically evaluate, and write atomic checkpoints to `train.out_dir` (e.g., `./runs/mnist/step000000000.pt` and `latest.pt`).


## Configuration

Configs are OmegaConf-first with structured dataclass defaults. Defaults fill missing keys, user YAML overrides defaults, and CLI overrides (if any) apply last.
Use OmegaConf interpolation for cross-section reuse (e.g., `logging.csv_path: ${train.out_dir}/log.csv`).

By default, Koochak enforces strict configuration to minimize surprises.

- Strict mode: unknown YAML keys cause an immediate error before training. To relax, set `train.strict_config: false`.
- Warnings: if strict is disabled, unknown keys print rank-0 warnings when `train.config_warn_unknown: true` (default true).

Example YAML toggles:

```
train:
  strict_config: true       # default
  config_warn_unknown: true # default, applies when strict_config=false
```

At startup, a brief config summary prints sections present, unknown keys (if any), and strict status.

Canonical template: `examples/config_template.yaml`.

In code, use `koochak.config.load_config(path)` and `koochak.config.get_section(cfg, "train")` (or similar) to access sections.

DDP sharding is explicit and opt-in:

- `train.shard_dataset: true` with `train.shard_dataset_mode: iterable|map` to shard the training dataset.
- `train.shard_eval_dataset: true` with `train.shard_eval_dataset_mode: iterable|map` to shard the eval dataset.
- `train.warn_unsharded: false` to disable rank-0 warnings when DDP runs without Koochak sharding.

You can also shard manually in custom code via `koochak.data.sharding.shard_dataset(...)`.

## Config Map

- `train` – consumed by `koochak/loop.py` and utilities (device, DDP/sharding, logging cadence, checkpoints, EMA, AMP).
- `data` – consumed by examples or your dataset builders; use OmegaConf interpolation for shared values.
- `optim` – consumed by `koochak/optim/build.py` for optimizer + scheduler construction.
- `logging` – consumed by CLI/examples to configure stdout/CSV/JSONL hooks.
- `wandb` – consumed by `koochak/logging/wandb_logger.py`.
- `entry` – consumed by `koochak/cli/train.py` to import user callables.



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
morbo:
  enabled: false
  project_id: your-project
  socket_path: /tmp/morbo-agent.sock
  max_weight_sample_values: 256
```

The CLI loads config via `koochak.config.load_config`, prints the summary, builds the optimizer/scheduler from `optim`, attaches stdout/CSV/JSONL/W&B hooks, optionally attaches Morbo hooks when `morbo.enabled` is true, resumes from the latest checkpoint under `train.out_dir`, and calls `training_loop` with `train_cfg`. Morbo keeps a stable logical `run_id` in `<train.out_dir>/.morbo-identity.json` and creates a new `attempt_id` for each launch unless one is supplied explicitly. The resolved IDs are also included in the serialized training config stored in checkpoints.


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
  train_cfg: Mapping[str, Any],
  config_json: Optional[Mapping[str, Any]] = None,
  checkpoint_dict: Optional[Dict[str, Any]] = None,
  eval_dataset: Optional[Iterable] = None,
  eval_fn: Optional[Callable] = None,
  hooks: Optional[Dict[str, list[Callable]]] = None,
) -> Dict[str, Any]
```

- `step_fn(model, batch, ctx)` returns `{"loss": Tensor, ...}`; any additional scalar values are logged.
- `ctx` contains `device`, `rank/world_size`, `autocast`, `scaler`, `config_json`, and `train_cfg`.
- The loop handles gradient accumulation, AMP, optional grad clipping, scheduler stepping (per `train.scheduler_step`), evaluation hooks, automatic DDP bootstrap/wrapping when `train.ddp` is true, and deterministic checkpointing.
- Rank-0 prints a compact parameter count banner at startup to highlight model size changes.
- Non-finite gradients are zeroed and skipped with a rank-0 warning instead of crashing the run.
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
- Hook events emitted by the loop include `on_train_start`, `on_step_end`, `on_log`, `on_eval_end`, `on_checkpoint`, `on_train_end`, and `on_exception`.
- Built-in hooks:
  - `koochak.logging.stdout.make_stdout_hooks()` – TSV prints; rank-0 only.
  - `koochak.logging.csv.make_csv_hooks(path)` – append metrics to CSV; rank-0 only.
  - `koochak.logging.jsonl.make_jsonl_hooks(path)` – one JSON per line; rank-0 only.
  - `koochak.logging.wandb_logger.make_wandb_hooks(cfg)` – W&B logging/artifacts; rank-0 only.
- Each built-in logger records the resolved config once at `on_train_start` (stdout prints JSON, CSV/JSONL write a config row, W&B receives it via `wandb.init(config=...)`).
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

W&B artifacts:
- The W&B hook versions checkpoints as a single artifact per run named `<prefix>-<run_id>` (default prefix `model`).
- Each upload includes aliases: `latest`, `step-<n>`, and when improved metrics are seen, `best` and `best-<metric>`.
- Config overrides (optional) under `wandb`:
  - `artifact_name_prefix` (str, default `model`)
  - `artifact_type` (str, default `model`)

## EMA Support

- Enable EMA by setting `train.ema.enabled: true` (or by providing `decay`/`profile` keys while `enabled` is unset). Nested config lives under `train.ema.*`; legacy flat keys (`ema_decay`, `ema_eval`, etc.) are still honored.
- Supported options: `decay`, `decay_init`, `warmup_steps`, `schedule` (`constant`, `linear`, `cosine`), `profile` (`constant`, `power`), `gamma`/`srel` for power-law schedules, `offload_to_cpu`, and `eval_with_ema` to run eval with shadow weights.
- Dual EMA tracking is available via `train.ema.dual.enabled` plus `gamma1/gamma2` or `srel1/srel2`; both shadows are saved and restored from checkpoints.
- EMA state is serialized alongside the model (and matches state-dict prefixes automatically) so resumes and manual loads stay seamless.

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

- When `train.ddp: true`, the loop auto-initializes the process group (if needed), pins the model to the local device, and wraps it in `torch.nn.parallel.DistributedDataParallel`. Pass `train.find_unused_parameters: true` if you need the corresponding DDP flag.
- Sharding is explicit: use `train.shard_dataset`/`train.shard_dataset_mode` (and the eval equivalents) or call `koochak.data.sharding.shard_dataset(...)` in custom code. If DDP is enabled and datasets are not marked as sharded, rank 0 emits a warning by default.
- If you prefer manual control, initialize ahead of time via `koochak.core.dist.init_process_group(...)`; the loop will detect the existing group and skip auto-init.
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
- On resume, the training loop restores RNG state from the checkpoint (Python/NumPy/Torch CPU/CUDA) before resuming steps, so randomness inside `step_fn` (e.g., `torch.rand`) is reproducible across restarts.
- In DDP, prefer per-rank seeding (e.g., `set_all_seeds(seed + rank)`) and rank-aware worker seeding (`make_worker_init_fn(seed, rank=rank)`) to avoid correlated randomness. Checkpoints saved on rank 0 include per-rank RNG states and are used on resume to restore each rank’s RNG deterministically.

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
