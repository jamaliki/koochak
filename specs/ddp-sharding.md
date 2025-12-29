# DDP Dataset Sharding

## Summary
Define an opt-in, explicit dataset sharding helper for DDP runs, with minimal loop/CLI integration and a default warning when DDP runs without Koochak sharding.

## Context
- Today, `koochak/data/sharding.py` exposes `shard_iterable`, but the loop does not apply it.
- Users can forget to shard in DDP, causing each rank to replay the same data.
- Auto-detection (e.g., DistributedSampler) is out of scope; sharding should be explicit and simple.

## Goals
- Provide a simple, explicit way to shard both iterable and map-style datasets.
- Allow opt-in sharding via config in loop/CLI, using the same helper.
- Avoid double-sharding if both user code and config apply sharding.
- Emit a rank-0 warning by default when DDP runs without Koochak sharding.

## Non-Goals
- Auto-detecting existing sharding or sampler usage.
- Handling DataLoader-specific behavior or sampler detection.
- Enforcing sharding by default in DDP.

## Requirements
- Functional:
  - Add explicit helpers for iterable and map-style dataset sharding.
  - Provide a single entry helper callable from user code and CLI/loop.
  - Add a sharding marker so repeated application is idempotent.
  - Add config toggles to opt-in and to warn when sharding is absent.
- Non-functional:
  - Keep the API simple and minimal.
  - Preserve performance with stride-based sharding.
  - Maintain compatibility with the current high-level API.

## Proposed Options
### Option A: Helper + opt-in config (recommended)
- Description: Add a sharding helper (iterable/map) with a marker. Loop/CLI call it when `train.shard_dataset: true` and `train.shard_dataset_mode` is set. Emit warning when DDP and no sharding marker, unless disabled.
- Pros: Simple explicit control, works for both CLI and custom loops, no auto-detect, idempotent.
- Cons: Requires user to choose a mode; warning can be ignored.
- Risks: Misconfigured mode could shard incorrectly.

### Option B: Helper only
- Description: Provide helpers but do not integrate with loop/CLI. Only warn in DDP when not sharded.
- Pros: Zero API changes in loop/CLI.
- Cons: Lower discoverability; higher chance users miss the helper.
- Risks: Continued throughput loss if users ignore warnings.

## Decision
Choose Option A. It keeps sharding explicit and simple, supports both CLI and custom loops, and meets the requirement to avoid auto-detection while offering a default warning and idempotent behavior.

## Detailed Design
### Helpers
Add a small set of helpers in `koochak/data/sharding.py`:
- `is_sharded(obj) -> bool`: Returns True if a dataset is already sharded by Koochak (marker attribute).
- `mark_sharded(obj)`: Sets a marker attribute (e.g., `__koochak_sharded__ = True`).
- `shard_iterable_dataset(iterable, rank, world_size)`: Returns a `ShardedIterable` wrapper that yields items where index % world_size == rank. If already sharded, returns as-is.
- `shard_map_dataset(dataset, rank, world_size)`: Returns a `ShardedMapDataset` wrapper with:
  - `__len__`: `ceil((len(dataset) - rank) / world_size)` (standard stride count)
  - `__getitem__(i)`: `dataset[i * world_size + rank]`
  - Raises `IndexError` if `i * world_size + rank >= len(dataset)`
  - If already sharded, returns as-is.
- `shard_dataset(dataset, *, rank, world_size, mode)`: Dispatches to iterable or map helpers. `mode` is required (no auto-detect).

Notes:
- Replace the existing `shard_iterable` with `shard_dataset` to avoid duplicate APIs and to enforce explicit mode selection.
- The wrappers should be light and only implement required dataset interfaces.

### Loop / CLI integration
- Add config keys:
  - `train.shard_dataset: bool` (default: false)
  - `train.shard_dataset_mode: "iterable" | "map"` (required when `train.shard_dataset: true`)
  - `train.shard_eval_dataset: bool` (default: false)
  - `train.shard_eval_dataset_mode: "iterable" | "map"` (required when `train.shard_eval_dataset: true`)
  - `train.warn_unsharded: bool` (default: true)
- When `config.ddp` is true and `train.shard_dataset` is true, call `shard_dataset(..., mode=train.shard_dataset_mode)` on the train dataset used by the loop/CLI.
- When `config.ddp` is true and `train.shard_eval_dataset` is true, call `shard_dataset(..., mode=train.shard_eval_dataset_mode)` on the eval dataset used by the loop/CLI.
- If `config.ddp` is true and `train.warn_unsharded` is true, emit a rank-0 warning when `is_sharded(dataset)` is false (do this for both train and eval when provided).
- Ensure idempotency: calling the helper twice returns the same dataset or a no-op wrapper.

### Error handling
- If `train.shard_dataset` is true and `train.shard_dataset_mode` is missing or invalid, raise a clear `ValueError` during startup.
- If `train.shard_eval_dataset` is true and `train.shard_eval_dataset_mode` is missing or invalid, raise a clear `ValueError` during startup.

### Documentation
- Add a brief README note explaining:
  - Explicit sharding requirement in DDP.
  - `train.shard_dataset`/`train.shard_dataset_mode` usage.
  - `train.shard_eval_dataset`/`train.shard_eval_dataset_mode` usage.
  - How to call the helper directly in custom code.
  - Meaning of the warning and how to disable it.

## Rollout / Migration
- Default behavior remains unchanged: no sharding unless opted in.
- New warnings may appear in DDP runs if sharding is not used; users can disable with `train.warn_unsharded: false`.

## Observability
- Log a rank-0 warning when DDP is enabled but dataset is not marked as sharded.
- Success criteria: reduced accidental full-duplicate data passes in DDP.

## Testing Plan
- Unit tests for `shard_iterable_dataset`:
  - World size 2 and 3, verify deterministic striding per rank.
  - Idempotency when called twice.
- Unit tests for `shard_map_dataset`:
  - Correct `__len__` and indexing for multiple world sizes.
  - Idempotency when called twice.
- Warning test:
  - When DDP enabled and dataset unsharded, warning is emitted (rank 0 only).

## Risks and Mitigations
- Risk: Users choose the wrong `mode`.
  - Mitigation: clear documentation and startup error on missing mode.
- Risk: Users ignore warnings.
  - Mitigation: warning is default-on and prominent; documentation highlights impact.

## Open Questions
- None.
