# Koochak – Agent Notes and Design Updates

This doc tracks incremental design decisions and changes from the initial design_doc.md as we implement the stack. New contributors: please read README.md first to get oriented, then come back here for active notes and TODOs. Always update README.md and this TODO list whenever behavior, layout, or usage changes.

## Implemented so far

- Core loop
  - `koochak/loop.py` implements the function-first `training_loop(...)` with AMP, grad accumulation, grad clipping, eval hooks, and checkpointing. Loop returns a resume-ready checkpoint dict.
  - Loop uses small, focused helpers for precision, config, device, RNG state, sharding, and hooks dispatch.

- Hooks
  - `koochak/core/hooks.py` provides `merge`, `add`, and `emit` utilities.
  - Rank-0 gating: provided through helpers (see below) and applied to built-in hooks so logging is emitted once in DDP.

- Precision
  - `koochak/core/precision.py` exposes `autocast_context(mode, device)` and `Scaler(mode)` (no-op except for fp16).

- Distributed
  - `koochak/core/dist.py` adds `init_process_group`, `barrier`, `rank`, `world_size`, and `rank0` helpers.
  - `koochak/data/sharding.py` provides `shard_iterable(iterable, rank, world_size)` for deterministic striding on IterableDataset-like sources.
  - The loop shards the dataset when `config.ddp=True`, and places `barrier()` calls around checkpointing. Only rank 0 writes checkpoints.

- Storage
  - `koochak/storage/checkpoint.py` – atomic save/load, `latest(dir)`, and `best(dir, key)`; maintains a `latest.pt` pointer and prunes with `keep_last_k`.

- Logging
  - `koochak/logging/stdout.py` – compact TSV stdout logger + `make_stdout_hooks()`.
  - `koochak/logging/wandb_logger.py` – lazy-import W&B hooks, logs metrics and artifacts, tracks `best/*` summaries.

- Optim
  - `koochak/optim/build.py` – tiny builders for optimizers (AdamW/Adam/SGD) and schedulers (cosine, step, plateau). Added cosine-with-warmup (`cosine_warmup`).

- Example
  - `examples/mnist/` – YAML-driven MNIST trainer (`config.yaml`, `main.py`). Minimal CLI (`--config` only). Uses stdout hooks by default; optional W&B via YAML.

## Deviations or clarifications

- Package layout: We keep training loop under `code/loop.py` for clarity during early development, with helpers under `koochak/*` per design. We can relocate `code/loop.py` to `koochak/core/loop.py` later without behavioral changes.
- Hook gating: The loop gates some events by rank 0; we also add rank-0 gating utilities and apply them inside built-in hooks so user code can safely emit hooks from any rank if desired.
- Atomic/fs/pruning helpers currently live in `storage/checkpoint.py` for simplicity; will be split into `storage/atomic.py`, `storage/pruning.py`, and `storage/fs.py` as a follow-up.

## Next up

- Add rank-0 gating helpers and update built-in hooks to use them.
- Add CSV and JSONL loggers with simple hook factories.
- Add `utils/stats.py` (SmoothedMeter, Throughput, EMA).
- Optional: stats in stdout formatting, and JSON Lines schema.

## TODO (keep this list up-to-date)

This list guides ongoing work. All contributors (agents and humans) should update it as tasks are added/completed.

- Split storage helpers into `storage.atomic`, `storage.pruning`, and `storage.fs` [DONE]
- Add `utils/timeit.py` scoped timers [DONE]
- Rank-0 gating helper and apply to built-in hooks [DONE]
- CSV and JSONL loggers [DONE]
- Wire CSV/JSONL logging via YAML and attach hooks in example [DONE]
- Stats utils: SmoothedMeter, Throughput, EMA [DONE]
- Refine stdout formatting to optionally include smoothed stats [TODO]
- Extract remaining storage helpers (e.g., artifact naming) if needed [TODO]
- Add `data/take.py` helper or extend iterable utils with `take(n)` [DONE]
- DDP checkpoint compatibility [TODO]
  - Problem: When the model is wrapped in `DistributedDataParallel`, `model.state_dict()` includes `module.` prefixes, which complicates resuming across DDP vs single-GPU. Our loop currently saves `model.state_dict()` directly.
  - Plan:
    1) Save the underlying module weights: if `hasattr(model, "module")`, save `model.module.state_dict()`; otherwise `model.state_dict()`.
    2) Provide small load helpers to add/remove `module.` prefixes as needed when users load a checkpoint.
    3) Document best practices in README (resuming with/without DDP).
- DDP convenience: example torchrun entry that calls `dist.init_process_group` [DONE]
- Optional: cli/train.py thin wrapper around `training_loop` [TODO]
  - Purpose: Generic CLI to run training from YAML without custom scripts.
  - Inputs: YAML pointing to Python callables (module:function) for `model_fn`, `dataset_fn` or `dataloader_fn`, `step_fn`, optional `eval_fn`. Also uses `train`/`optim`/`logging`/`wandb` sections.
  - Behavior: Imports callables, builds model/optim/scheduler/datasets, attaches hooks (stdout/CSV/JSONL/W&B), handles resume via latest checkpoint, supports DDP (auto `init_process_group` + sharding), then calls `training_loop`.
  - Value: Consistent UX; reduces boilerplate; easier automation.
- Tests: unit tests for precision helpers, checkpoint round-trip, shard_iterable [TODO]

Note: Keep this TODO section synchronized with the codebase state and design decisions.
