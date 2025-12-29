Throughput Lessons for Koochak Training Loop
============================================

This note captures throughput-impacting behaviors observed in `koochak/loop.py`
and explains why they matter, when they are likely to show up, and how to
mitigate them. It is meant to guide future changes and reviews.

1) Per-microstep loss scalar sync
---------------------------------
What happens:
- Inside the grad-accum loop, the code does:
  `total_loss_scalar += float(loss.detach())`.
- `float(tensor)` on CUDA forces a device-to-host sync.

Why this hurts:
- With grad accumulation, this can trigger a sync per micro-batch.
- Syncs serialize the CPU with GPU work and degrade throughput, especially
  in small models where the compute is short and sync overhead dominates.

Mitigation:
- Keep the running loss on device and only convert to a Python float on
  logging steps (e.g., `loss_sum += loss.detach()` and `.item()` when logging).
- Or store a tensor on CPU only at log cadence (or for rank 0 only).

Impact scope:
- Affects all users, worse with grad_accum > 1 and frequent logging.


2) DDP dataset sharding not enforced in loop
--------------------------------------------
What happens:
- The loop imports `shard_iterable` but does not apply it.
- If the caller does not shard the dataset, each DDP rank iterates the same
  data, so compute is duplicated across ranks.

Why this hurts:
- Effective throughput per unique sample plummets because each rank repeats
  work; it also wastes bandwidth and compute.
- The model "trains slower" in terms of progress per GPU-hour.

Mitigation:
- Apply `shard_iterable` inside the loop when `ddp_enabled` is true and the
  dataset is an iterable (or provide an explicit opt-out).
- Alternatively, enforce that DDP entrypoints always shard and add checks to
  catch unsharded iterables.

Impact scope:
- DDP training; serious if the dataset is iterable (not a DistributedSampler).


3) Checkpointing stalls other ranks
-----------------------------------
What happens:
- Only rank 0 checkpoints; other ranks keep training until the next all-reduce.
- When rank 0 is in disk IO, other ranks wait at sync points.

Why this hurts:
- Periodic IO can create throughput cliffs at checkpoint intervals.
- The larger the model and the slower the storage, the worse the stall.

Mitigation:
- Add a barrier before/after checkpointing to align ranks, or
- Offload checkpoint write to a background thread/process, or
- Throttle checkpoint frequency and use smaller `keep_last_k`.

Impact scope:
- DDP training, especially on shared or slow filesystems.


4) Non-finite gradient checks are expensive
-------------------------------------------
What happens:
- The loop scans all parameters and calls `torch.isfinite(...).all()`.
- This can be a full pass over parameters and may sync to CPU.

Why this hurts:
- For large models, this is a sizable overhead on any step that triggers it.
- The overhead grows with model size and tight iteration time.

Mitigation:
- Reduce `nonfinite_grad_check_every` frequency or disable in stable regimes.
- Narrow the check to a subset of parameters or hook into the scaler overflow
  path where possible.

Impact scope:
- Any run that enables `nonfinite_grad_check_every`.


Notes
-----
- These are performance concerns, not correctness bugs. They should be
  considered when optimizing throughput or scaling to multi-GPU.
- See `analysis/ddp_sharding_plan.md` for a more detailed plan around DDP
  sharding policies and their tradeoffs.
