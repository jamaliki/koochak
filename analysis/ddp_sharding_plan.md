DDP Dataset Sharding Plan
========================

Goal
----
Define a clear, safe policy for dataset sharding in DDP so that each rank
processes a unique subset of data without forcing users into one rigid pattern.

Current Behavior (Loop)
-----------------------
- `koochak/loop.py` imports `shard_iterable` but does not apply it.
- This means DDP sharding only happens if the caller does it externally.

Problem
-------
- It is easy to forget to shard when using IterableDataset-like sources.
- In DDP, unsharded iterables cause each rank to repeat the same data,
  effectively wasting compute and lowering throughput per unique sample.

Constraints
-----------
- The loop accepts any iterable. Some users may pass DataLoader instances
  with DistributedSampler, where manual sharding would be incorrect.
- The loop should remain simple and avoid surprising the caller.
- There should be a safe, explicit opt-out or detection for "already sharded".

Options
-------
Option A: Loop-enforced sharding (default-on for DDP)
- Behavior:
  - If `ddp_enabled` and `world_size > 1`, wrap dataset with `shard_iterable`.
  - Allow opt-out via `train.shard_iterable: false`.
- Pros:
  - Safe by default for iterables.
  - DDP throughput is protected for common cases.
- Cons:
  - Can double-shard if caller passes a DataLoader with DistributedSampler.
  - Requires detection or opt-out to avoid incorrect behavior.

Option B: Explicit-only (status quo + safety warning)
- Behavior:
  - Do not shard inside the loop.
  - If DDP is enabled and dataset is an iterable, emit a rank-0 warning
    recommending sharding unless `train.assume_sharded: true`.
- Pros:
  - No risk of double-sharding.
  - Keeps loop minimal and caller-controlled.
- Cons:
  - Easy to ignore warnings.
  - Still allows silent throughput loss.

Option C: Detect common "already sharded" cases
- Behavior:
  - In DDP, if dataset is a DataLoader with a DistributedSampler, do not shard.
  - Otherwise, apply `shard_iterable` (or warn if `train.shard_iterable` is unset).
- Pros:
  - Better default behavior without forcing opt-outs.
  - Covers the most common DDP pattern (DataLoader + DistributedSampler).
- Cons:
  - Detection can be brittle (custom samplers or loaders).
  - More logic in the loop; still might mis-detect edge cases.

Option D: Shard via entrypoint helpers only
- Behavior:
  - Provide a helper (or wrapper) in examples/cli that always shards DDP
    iterables and documents the requirement.
- Pros:
  - Keeps core loop minimal.
  - Enforces best practices in common paths (CLI, examples).
- Cons:
  - Custom integrations still risk unsharded datasets.
  - Throughput risk remains for users bypassing helpers.

Recommendations to Decide
-------------------------
1) Decide which user experience is preferred:
   - Safe-by-default inside the loop (Option A/C), or
   - Explicit control with warnings (Option B/D).
2) If safe-by-default, pick how to avoid double-sharding:
   - Add `train.shard_iterable` toggle, and/or
   - Detect DistributedSampler in DataLoader.
3) Document the behavior clearly in README and in examples.

Proposed Schema / Config Keys (if needed)
-----------------------------------------
- `train.shard_iterable: bool` (default: true when DDP)
- `train.assume_sharded: bool` (default: false)
- `train.warn_unsharded: bool` (default: true)

Testing Plan
------------
- Unit test: `shard_iterable` applies in DDP and produces correct striding.
- Integration test: DataLoader with DistributedSampler is not double-sharded.
- Regression test: non-DDP runs unchanged.

Open Questions
--------------
- Should sharding live in the loop or only in CLI/examples?
- Do we need to support both iterable and map-style datasets explicitly?
- Should we detect or allow "already sharded" iterables via a marker attribute?
