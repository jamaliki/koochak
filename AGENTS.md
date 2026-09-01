# Koochak – Agent Notes and Design Updates

This doc tracks incremental design decisions and changes from the initial design_doc.md as we implement the stack. New contributors: please read README.md first to get oriented, then come back here for active notes and TODOs. Always update README.md and this TODO list whenever behavior, layout, or usage changes.

## Important things to keep in mind

- The philosophy is *functions first*
- Do not catch exceptions unless necessary, fail *fast and loud* if an unexpected error occurs. If we are expected to have some errors, those should be caught explicitly and not silently.

## Implemented so far

- Core loop
  - `koochak/loop.py` implements the function-first `training_loop(...)` with AMP, grad accumulation, grad clipping, auto DDP bootstrap/wrapping, eval hooks, EMA (single + dual) tracking, and deterministic checkpointing. The loop atomically saves its terminal in-memory state before completion hooks and returns the same resume-ready checkpoint dict.
  - Loop uses small, focused helpers for precision, config, device, RNG state (including per-rank gather/restore), sharding, and hooks dispatch, and emits a rank-0 parameter-count banner plus warnings when gradients become non-finite.

- Hooks
  - `koochak/core/hooks.py` provides `merge`, `add`, and `emit` utilities.
  - Rank-0 gating: provided through helpers (see below) and applied to built-in hooks so logging is emitted once in DDP.

- Precision
  - `koochak/core/precision.py` exposes `autocast_context(mode, device)` and `Scaler(mode)` (no-op except for fp16).

- Distributed
  - `koochak/core/dist.py` adds `init_process_group`, `barrier`, `rank`, `world_size`, and `rank0` helpers.
- `koochak/data/sharding.py` provides `shard_dataset(..., mode=...)`, `shard_iterable_dataset`, and `shard_map_dataset` for explicit DDP sharding.
  - The loop shards the dataset when `config.ddp=True`, and places `barrier()` calls around checkpointing. Only rank 0 writes checkpoints.

- Storage
  - `koochak/storage/checkpoint.py` – atomic save/load, publication manifests,
    `latest(dir)`, and `best(dir, key)`; maintains a `latest.pt` convenience
    pointer and prunes checkpoint/manifest pairs with `keep_last_k`.

- Logging
  - `koochak/logging/stdout.py` – compact TSV stdout logger + `make_stdout_hooks()`.
  - `koochak/logging/wandb_logger.py` – lazy-import W&B hooks, logs metrics and artifacts, tracks `best/*` summaries.
  - `koochak/logging/events.py` – bounded, rank-0 lifecycle/progress events plus
    a lazy optional Scruffy adapter. Numbered checkpoints publish strict,
    deterministic artifact evidence after their ready manifest exists; full
    config objects and checkpoint contents never enter coordination events.
    Scruffy checkpoint acknowledgement is opt-in and synchronous only for
    strict artifact events; configured rejection/timeout fails closed, while
    the default telemetry path remains asynchronous and non-fatal.
  - Stdout and W&B record resolved config at `on_train_start`; CSV/JSONL remain metric logs.

- Jobs
  - `koochak/jobs` compiles strict environment YAML and resolved training config into an immutable launch manifest.
  - One isolated runner verifies the environment and uses exact Python argv for both Pazuzu and Scruffy. Thin adapters retain each backend's native resource model.
  - Agent-authored experiment submission must live in a committed Python script using `prepare_run`, then `submit_pazuzu` or `submit_scruffy`. Do not assemble SSH, `sbatch`, or scheduler CLI submissions in tool calls.
  - `submit_scruffy(..., wait_for=[...])` declares intermediate artifact gates.
    Use the numbered checkpoint artifact ID and never `latest.pt`.
  - Profiles contain no secret values. They name required secrets, inherit only Koochak's fixed runtime identity, and cannot override it. All ordinary values, including compiler and NCCL settings, belong under `environment.set`.

- Optim
  - `koochak/optim/build.py` – tiny builders for optimizers (AdamW/Adam/SGD) and schedulers (cosine, step, plateau). Added cosine-with-warmup (`cosine_warmup`).

- Example
  - `examples/mnist/` – YAML-driven MNIST trainer (`config.yaml`, `main.py`). Minimal CLI (`--config` only). Uses stdout hooks by default; optional W&B via YAML.

## Deviations or clarifications

- Hook gating: The loop gates some events by rank 0; we also add rank-0 gating utilities and apply them inside built-in hooks so user code can safely emit hooks from any rank if desired.
- Storage helpers now live in `storage.atomic`, `storage.pruning`, `storage.fs`, and `storage.naming`; checkpoint orchestrates them for atomic writes and pruning.
- The loop auto-initializes a process group (when unset) and wraps models in DDP when `config.ddp=True`; set `config.find_unused_parameters` to forward the DDP flag.

## Next up

- Refine stdout formatting to surface optional smoothed stats and throughput.
- Extract remaining storage helpers (artifact naming, etc.) if further splitting proves useful.
- Ship the thin `cli/train.py` wrapper once schema/entry helpers stabilize.
- Broaden unit test coverage (resume determinism, scheduler-on-eval policy, grad-accum equivalence).

## TODO (keep this list up-to-date)

This list guides ongoing work. All contributors (agents and humans) should update it as tasks are added/completed.

- Split storage helpers into `storage.atomic`, `storage.pruning`, and `storage.fs` [DONE]
- Add `utils/timeit.py` scoped timers [DONE]
- Rank-0 gating helper and apply to built-in hooks [DONE]
- CSV and JSONL loggers [DONE]
- Wire CSV/JSONL logging via YAML and attach hooks in example [DONE]
- Stats utils: SmoothedMeter, Throughput, EMA [DONE]
- Config system overhaul (OmegaConf structured defaults + section-based loader) [DONE]
- Refine stdout formatting to optionally include smoothed stats [TODO]
- Extract remaining storage helpers (e.g., artifact naming) if needed [TODO]
- Add `data/take.py` helper or extend iterable utils with `take(n)` [DONE]
- DDP checkpoint compatibility [DONE]
  - Loop saves underlying module weights when present, gathers per-rank RNG, and `checkpoint.match_state_dict_to_model` adapts prefixes for resume across DDP/non-DDP. README documents the flow.
- DDP convenience: example torchrun entry that calls `dist.init_process_group` [DONE]
- DDP dataset sharding helpers + config toggles + warnings [DONE]
- Optional: cli/train.py thin wrapper around `training_loop` [TODO]
  - Purpose: Generic CLI to run training from YAML without custom scripts.
  - Inputs: YAML pointing to Python callables (module:function) for `model_fn`, `dataset_fn` or `dataloader_fn`, `step_fn`, optional `eval_fn`. Also uses `train`/`optim`/`logging`/`wandb` sections.
  - Behavior: Imports callables, builds model/optim/scheduler/datasets, attaches hooks (stdout/CSV/JSONL/W&B), handles resume via latest checkpoint, supports DDP (auto `init_process_group` + sharding), then calls `training_loop`.
  - Value: Consistent UX; reduces boilerplate; easier automation.
- Tests: unit tests for precision helpers, checkpoint round-trip, sharding helpers [TODO]
- Reproducible jobs API [DONE]
  - Strict, site-neutral environment profiles with no ambient inheritance or private cluster policy.
  - Content-addressed manifests and resolved configs are staged atomically and cannot be clobbered by another agent.
  - The runner checks interpreters, files, packages, CUDA, the C compiler, and Torch compilation before user code.
  - Python-only Pazuzu and Scruffy adapters replace the raw SSH backend.
  - Tests cover validation, digesting, tamper detection, environment ownership, preflight records, adapters, and private-string scanning.
- Phase 4 workflow foundations [DONE]
  - Immutable `PreparedTask`/`PreparedWorkflow` models and all-or-none `submit_scruffy_workflow` staging.
  - Immutable file/directory ready manifests with deterministic file ordering, byte size, SHA256, provenance, and counts.
  - Declared output validation/publication in a managed child process group; SIGUSR1 is never sent to the parent allocation, surviving descendants fail closed, and partial event retries use deterministic IDs.
  - Shared-storage race checks use no-follow descriptors and no-overwrite links but assume cooperative same-user writers; use isolation for hostile same-UID workloads.
  - Final-pin gate: after Scruffy stabilizes, update the optional dependency to its reviewed commit and rerun both repositories' full suites before release or deployment.
  - `storage.checkpoint.resolve_auto_resume()` returns the validated loaded payload alongside its path so callers can construct datasets from `next_step` before entering `training_loop`; preloaded selection retains `auto_resume_selected` and artifact republishing.

Note: Keep this TODO section synchronized with the codebase state and design decisions. Prefer the authoritative "Open TODOs" section at the end.

Updates:
- DDP checkpoint compatibility implemented: loop saves `model.module.state_dict()` when present; added helpers `match_state_dict_to_model`, `strip_module_prefix`, `add_module_prefix` and README instructions for resuming across DDP/non-DDP.
- RNG restore on resume implemented: loop restores RNG from checkpoint (`utils.seed.set_rng_state`), tests added.
- Dual power-EMA tracking restored: `train.ema.dual.enabled` now builds two power-profile EMA trackers, updates/restores them with the loop, and saves them under `checkpoint["ema_dual"]`. `ema_posthoc.reconstruct_power_ema_state_dict(...)` combines dual EMA states from multiple saved checkpoints for EDM2-style post-hoc tuning.

New design TODOs

- Config system overhaul (OmegaConf structured defaults + section-based loader + schema-based unknown-key summary) [DONE]
  - New `koochak/config.py` with dataclasses, `load_config`, `summarize`, and `get_section`.
  - Legacy usage-tracking/unused-key detection retired in favor of schema-based unknown-key checks.


CLI enhancements (design TODOs)

- Expand CLI to accept simple kwargs [DESIGN TODO]
  - Goal: Allow passing positional/keyword args to entry callables via YAML without writing custom glue code.
  - YAML shape (examples):
    entry:
      model: your_pkg.model_defs:make_model
      model_args: [256, 10]                # optional positional args
      model_kwargs: {dropout: 0.1}         # optional keyword args
      dataset: your_pkg.data:train_dataset
      dataset_kwargs: {data_dir: ./data, batch_size: ${data.batch_size}}
      step: your_pkg.train:step_fn         # signature remains (model, batch, ctx)
      eval_dataset: your_pkg.data:val_dataset
      eval_dataset_kwargs: {data_dir: ./data, batch_size: ${data.batch_size}}
      eval_fn: your_pkg.train:eval_fn
  - Implementation notes:
    1) Use OmegaConf interpolation (already integrated) for `${section.key}` resolution.
    2) In koochak.cli.train, when constructing objects, retrieve *_args (list) and *_kwargs (dict) if present and pass them to the callable.
    3) Keep types as provided by YAML; OmegaConf preserves original types on resolve.
    4) Strict config still applies: extend schema to include entry.model_args, entry.model_kwargs, etc.

- Tiny helpers for safe importing [DESIGN TODO]
  - Purpose: Centralize import-from-string with robust errors and validation; reduce boilerplate and improve UX in CLI and examples.
  - Module: koochak/utils/imports.py (or entry.py)
  - API:
    - import_object(path: str, *, expect_callable: bool | None = None, expect_type: type | tuple[type, ...] | None = None) -> Any
      - Supports both "module:object" and "module.object" notations.
      - On ImportError/AttributeError, raise a ValueError with a friendly message showing the path, attempted module, and available attributes. Optionally suggest close matches via difflib.get_close_matches.
      - If expect_callable is True, verify callable(obj); if expect_type provided, isinstance check with a clear error.
    - call_with(obj, *args, **kwargs): wraps calling with a short context on exception (e.g., "error while calling entry.model(...)")
  - Optional niceties:
    - Signature introspection (inspect.signature) to warn when unexpected kwargs are provided.
    - A tiny dotted-path resolver util used by CLI to implement ${...} interpolation.
  - Tests: cover good/bad import strings, callable/type validation, and helpful error messages.

---

Status Update — 2025-08-31

Done recently
- Core loop tightened and modularized; DDP wrapping + sharding + checkpoint barriers.
- Storage split into `atomic`, `fs`, `pruning`; checkpoint uses atomic writes and pruning; latest/best helpers.
- Logging: stdout TSV + CSV + JSONL; W&B hooks; rank0 gating applied in hooks.
- Precision helpers (autocast + scaler), stats (SmoothedMeter/Throughput/EMA), and timeit utility.
- DDP-safe checkpoints: save underlying module weights; helpers to adapt state_dict keys; README guidance.
- RNG management: global seeding; rank-aware worker seeding; per‑rank RNG save/restore on resume.
- Data: `cycle`, `take`, and `shard_dataset`.
- Config system overhaul: structured defaults + section-based loader + schema-based unknown-key summary.
- Config: adopted OmegaConf for YAML loading and interpolation/resolution.
- Examples: MNIST (YAML-only); DDP launcher (`examples/mnist/ddp_main.py`).
- Generic CLI: `python -m koochak.cli.train --config ...` with strict validation and standard hooks.
- Generic CLI automatically attaches Scruffy workload-event hooks when both `SCRUFFY_ROOT` and `SCRUFFY_JOB_ID` identify a managed worker; Scruffy remains an optional lazy import.
- Safe evacuation is opt-in through `train.evacuation_enabled` or the `training_loop(..., evacuation=...)` API. `SIGUSR1` only sets an in-memory flag; after a completed optimizer update, DDP ranks reconcile the request, durably publish a numbered terminal checkpoint, cross the terminal barrier, publish its event, finish hooks, and exit with reserved code 75.
- `resume: auto` (the generic CLI default) selects only the highest numbered checkpoint with a valid ready manifest, exact path/size/SHA256, and valid resume cursor; `latest.pt` is never resume evidence.
- When auto-resume selects a checkpoint, the typed checkpoint artifact event is retried once through the configured event hook so transient Scruffy journal loss can heal without making ordinary progress telemetry fatal. Restartable W&B jobs must provide a stable `name` or `id` and set `resume: allow`; this is enforced only on attempts that actually resume.
- Tests: sharding, precision, checkpoint prefix adapt, checkpoint prune/save, checkpoint round‑trip, config validation, RNG restore.
- Tests: opt-in SIGUSR1 evacuation, terminal checkpoint publication ordering, strict numbered-checkpoint validation, and first-run-safe auto-resume.

Open TODOs (authoritative)
- CLI: support entry kwargs [TODO]
  - Extend schema to include `entry.model_args`, `entry.model_kwargs`, `entry.dataset_args/kwargs`, `entry.eval_dataset_args/kwargs`.
  - Use OmegaConf-resolved values; pass args/kwargs to entry callables in `koochak.cli.train`.
  - Add tests for args/kwargs plumbing.
- Safe imports helper [TODO]
  - `koochak/utils/imports.py` with `import_object` and `call_with` + friendly errors; tests.
- Refine stdout logging [TODO]
  - Optionally include smoothed stats (e.g., loss EMA/avg) and throughput in periodic prints.
- Storage niceties [DONE]
  - Added storage naming helpers (`koochak/storage/naming.py`) and improved W&B artifact integration: stable per-run artifact name with `latest`, `step-<n>`, and `best` aliases.
- More tests [TODO]
  - Resume determinism across partial runs (single/DDR), scheduler-on-eval policy, grad-accum equivalence.
- Documentation [TODO]
  - Add a short README for `examples/mnist` and a troubleshooting section for DDP (env vars, CUDA device binding).
