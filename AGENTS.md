# Project Agent Instructions

## Shell Safety

- In zsh commands and scripts, never use `path` as a variable or loop-variable
  name. In zsh, the special `path` array is tied to `PATH`, so assigning it
  overwrites the executable search path. Use names such as `file`, `relpath`,
  or `target_path` instead.

## Training Data I/O on Sandpit Tokyo

Lustre random I/O is extraordinarily slow relative to GPU training. Treat data
locality and cache residency as correctness requirements for experiment
throughput, not optional optimizations.

- Prefer one disjoint physical-shard ownership assignment across all ranks and
  DataLoader workers, followed by resident preload of every owned shard.
- Never infer resident-cache memory from a run that duplicated physical shards
  across workers. Fix and verify ownership first, then remeasure the unique
  decoded footprint.
- Do not introduce a small shard LRU as an unmeasured OOM hedge. A uniformly
  sampled cache of `k` shards from `N` owned shards has only approximately
  `k/N` hit rate and can turn training into repeated decompression from Lustre.
- Before changing `data.shard_cache_size` away from `null`, run a
  production-shaped gate with the real worker count, batch size, mixture,
  conditioning sidecars, compilation mode, and 240 GB cgroup. Record peak RSS,
  cache hit/miss counts, p50 and p90 step time, and finite-loss evidence.
- Reject a bounded cache unless that gate proves it is necessary for memory and
  does not materially regress warmed throughput or tail latency. Never choose
  an arbitrary capacity such as 8.
- Couple small per-shard conditioning data, including Progres sidecars, to the
  same disjoint ownership and preload policy. Avoid randomized small-file
  fan-out before the first batch.
- For the current L128 Atom14 mixture corpus, the measured unique decoded shard
  union is about 123.8 GiB. Production-shaped gates with all 576 owned shards
  resident peaked at about 137-139 GiB in a 240 GB cgroup and achieved roughly
  0.23 s p50 and 0.35 s p90 step time. Use resident caching unless a new
  measurement on a changed corpus or resource shape disproves this.
- Treat a cache hit rate near `cache_size / owned_shards` plus high step-time
  variability as a data-I/O regression. Diagnose cache counters and prefetch
  wait time before changing model or GPU settings.

Any production launcher that intentionally departs from these rules must carry
a machine-readable config diff and link to the representative gate evidence
that justified the departure.
