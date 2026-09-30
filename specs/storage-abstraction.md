# Storage Abstraction for Datasets and Checkpoints

## Summary
Put every durable byte Koochak reads or writes (dataset shards, checkpoints,
ready manifests) behind one narrow, object-store-shaped interface, so the same
training code scales across parallel filesystems, object stores, and FUSE
mounts over object storage. Site-specific backends plug in from private
packages; this repository stays site-neutral.

## Context
Koochak's storage layer assumes a local POSIX filesystem:

- `atomic_write` publishes with a temp file, `os.replace`, and a directory fsync.
- `write_immutable_file` publishes create-only artifacts with `os.link`.
- `checkpoint.save` maintains `latest.pt` as a symlink and falls back to a full
  copy of the checkpoint where symlinks are unsupported.
- `checkpoint.best` fully loads every checkpoint to read its metrics.
- Rank 0 serializes the whole checkpoint to one in-memory blob and training
  waits for the entire write.
- `ShardedIterable` splits samples across ranks after reading them, so every
  rank reads the full stream and discards most of it.

Probing a parallel filesystem and a FUSE mount over object storage (with a
caching tier) with `koochak.storage.probe`, from one 16-CPU compute node each,
showed behaviour that breaks several of these assumptions:

| Behaviour | Parallel filesystem | FUSE over object storage |
| --- | --- | --- |
| Exclusive create (`O_EXCL`) | yes | yes |
| Rename over an existing file | yes | yes (but not relied on) |
| Hard links | yes | no (`EPERM`) |
| Symlinks | yes | no |
| Append / in-place writes | yes | no (write-once files) |
| Reading a file still open for writing | sees bytes written | blocks ~20 s, then fails |
| First read after close matches the written bytes | yes | yes (no read-back delay seen) |
| Small-object put p50 / max | 8 ms / **137 s** | 0.9 s / 1.2 s |
| Small-object get / stat p50 | 0.5 ms / 0.2 ms | 225 ms / 96 ms |
| Listing 256 keys | 1.3 ms | **36 s** |
| 4 GiB write as 1 / 8 concurrent parts | 850 / 1030 MB/s | 355 / 455 MB/s |
| First read of just-written data, per stream | client-cached (GB/s) | **11–15 MB/s** |
| First read of a 4 GiB payload as 1 / 8 parts | client-cached | 14 / 103 MB/s |
| Cached dataset read, 1 / 16 worker processes | client-cached | 260 / 570 MB/s |
| Building an 8.5 GB shard dataset | 45 MB/s (write stalls) | 221 MB/s |

Parallel-filesystem reads were served largely from the writing node's page
cache, so they overstate cold reads there.

What this means for the design:

- **Per-request cost, not bandwidth, dominates small objects**, and listing is
  prohibitively slow on object storage. Training must read large shards named
  by an index and never list or open many small files.
- **Cold reads are limited per stream and scale with concurrency** (about
  8× for 8 streams). Readers need several shard downloads in flight, and
  checkpoints should be split into parts that resume reads in parallel.
- **Cached reads are about 20× faster than cold ones**, so the first epoch is
  the bottleneck. Warming the cache tier ahead of training, or staging each
  worker's shards to local memory or disk, pays off.
- **Read-back verification is unnecessary on this mount and expensive**
  (minutes for a multi-GB checkpoint read cold). Rely on manifest SHA256 checks
  at read time instead.
- **The parallel filesystem is fast when healthy but has multi-minute write
  stalls**, so checkpoint writes should not block training (async save).
- Anything that appends (loggers) or links (artifact manifests, `latest.pt`)
  cannot live on the object-storage mount.

## Goals
- One interface for durable bytes, with semantics every backend can honour.
- Datasets read per worker at file granularity, with no listing at startup and
  every byte verifiable.
- Checkpoints that commit with a manifest written last, can be written
  asynchronously and in parallel, and are selected by reading manifests only.
- Private backends without naming them in this repository.

## Non-Goals
- A general filesystem API (fsspec already exists; a backend may wrap it).
- Mutable state on object storage: logs, compiler caches, and W&B directories
  stay on POSIX scratch.
- A new dataset file format.

## Decision: write-once objects
The `Store` protocol (`koochak/storage/store.py`) exposes `get` (whole or byte
range), `open`, `put`, `stat`, `list(prefix)`, `delete`, and `local_path`.

- `put` is **create-only** (`FileExistsError` if the key exists) and returns
  only once other clients can read exactly those bytes.
- There is no rename, append, overwrite, or symlink. They cannot be atomic on
  object storage, and nothing Koochak persists needs them.
- Keys are relative, normalized, `/`-separated paths.

`LocalStore` implements it on POSIX, with settings chosen from probe results:

- `publish="link"` (default): hidden temp file plus hard link; readers never
  see a partial object.
- `publish="exclusive"`: `O_EXCL` create, write, close; for mounts without
  hard links that publish on close. A partial object is removed, never
  truncated.
- `fsync=False` for mounts that reject fsync.
- `verify_readback=True, settle_seconds=S` for mounts whose close completes
  asynchronously: re-read until size and SHA256 match, then fail loudly.

`open_store(location)` maps paths and `file://` URIs to `LocalStore`, and any
other `scheme://` through `register_store` or the `koochak.stores` entry-point
group. A private package typically registers a scheme whose factory returns a
`LocalStore` tuned for its mount, or a client for a native object API:

```toml
[project.entry-points."koochak.stores"]
mystore = "my_private_pkg.koochak_store:open_mystore"
```

## Datasets
Implemented in `koochak/data/shards.py`:

1. **Index.** `index.json` is written last and lists every shard's relative
   key, size, SHA256, and record count. Its SHA256 identifies the dataset
   version. Training reads the index instead of listing storage.
2. **Writer.** `ShardWriter` packs records into ~`target_bytes` shards using a
   `ShardFormat` (`encode`, `decode`, `trailer`). The default `TAR` format uses
   the WebDataset layout with deterministic headers, so reruns of an
   interrupted build reuse identical shards and refuse different ones.
3. **Plan.** `plan_shards` orders shards by `sha256(seed, epoch, key)` and gives
   each of the job's data-loading workers one contiguous run balanced by record
   count. It is pure and deterministic, needs no coordination, and fails
   identically on every worker when shards are too few or too uneven.

Next (phase 2):

- `ShardedStream(IterableDataset)`: read the plan for this worker, prefetch a
  bounded number of shards in a background thread, verify, decode, and apply a
  seeded shuffle buffer that resets at fixed shard windows. Each worker's
  stream never ends (epochs follow one another inside the worker), so DDP ranks
  cannot desynchronize on uneven shards; `max_steps` ends training.
- Resume from `next_step`: skip whole shards using record counts and replay at
  most one shuffle window. Record the index SHA256 and the loader layout (world
  size, workers, batch) in the checkpoint, and fail on mismatch unless the user
  opts into approximate repositioning.
- Staging: copy this node's planned shards to local disk (or RAM) with high
  concurrency, verifying SHA256, then read through `LocalStore`.
- An optional content-addressed local cache keyed by SHA256, and an optional
  "warm these keys" hint for backends with a cache tier.
- Size shards so per-request latency is negligible (hundreds of MB), and keep at
  least ~10 shards per data-loading worker for shuffle quality and balance.

## Checkpoints (phase 3)
A checkpoint becomes parts plus a manifest written last:

```
run/step000005000/model.pt
run/step000005000/optimizer.pt
run/step000005000/ema.pt
run/step000005000.ready.json   # create-only; step, next_step, metrics, per-part size+sha256
```

- **Async save.** Snapshot to pinned CPU memory synchronously, then serialize and
  `put` in the background, with at most one save in flight. `on_checkpoint` and
  strict acknowledgements fire when the manifest is committed. Terminal and
  evacuation saves drain the in-flight save and then save synchronously.
- **Parallel parts.** Parts upload concurrently, optionally spread across ranks,
  with rank 0 committing after it has every part record. Per-rank sharded state
  uses the same protocol.
- **Manifest-only selection.** Resume and `best` read manifests; `latest.pt`
  goes away (no symlink, no copy).
- **Pruning.** Delete the manifest first (uncommit), then the parts.
- **Replication.** Copying a committed checkpoint between stores (parts, then
  manifest) gives tiering, e.g. fast local disk first, then object storage.

## Artifacts (phase 4)
`storage.artifact.publish_artifact` uses hard links, which object-storage
mounts reject. Move ready manifests onto `Store.put`, and let the published
`path` become a store URI. Artifact gates that validate manifests must then
read through a `Store` too, which needs a coordinated protocol change in the
scheduler that consumes them.

## Testing Plan
- Store: round trips, byte ranges, create-only conflicts, streamed puts,
  cleanup of partial objects, prefix listing that hides in-progress writes,
  key validation, hard-link fallback error, read-back settling, URI
  resolution, registration, and entry-point discovery.
- Shards: tar determinism and compatibility with `tarfile`, writer rollover,
  commit-last, rerun idempotence, tamper detection, verification, and plan
  completeness, balance, determinism, and starvation errors.
- Probe: local run with cleanup, recommendation without hard links, CLI JSON.
- Phases 2 and 3 add resume-determinism tests across partial runs and
  mocked-mount tests for exclusive publish and delayed read-back.

## Risks and Open Questions
- Read-back verification doubles checkpoint I/O on mounts that need it. None
  was needed on the probed mount, where first reads always matched.
- Measurements so far come from CPU nodes; accelerator nodes may have a
  different network path. Cold multi-stream reads of data uploaded by other
  tools (rather than written through the mount) are not yet measured.
- The probe's concurrent-read phase mixes cached and uncached objects; phase 2
  should measure them separately and evict page caches before cold reads.
- Exact resume across a changed world size is not possible with rank-based
  plans; the default is to fail loudly.
- Whether node-local disks exist on the target compute nodes decides whether
  staging and tiered checkpoints are worth building.
