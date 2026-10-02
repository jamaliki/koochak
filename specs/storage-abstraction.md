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
| 512 MB read, 1 stream, cold / cached | 1029 / 2267 MB/s | **16 / 1017 MB/s** |
| 32 byte ranges of one 512 MB object, cold / cached | 3222 / 15371 MB/s | 127 / **155 MB/s** |
| 32 separate 512 MB objects read concurrently, cold | 904 MB/s | **356 MB/s** |
| 33 concurrent 512 MB writes | 274 MB/s | 840 MB/s |

The first parallel-filesystem read numbers were served largely from the
writing node's page cache; the 512 MB rows drop page caches first
(`posix_fadvise(DONTNEED)`) and are cold.

What this means for the design:

- **Per-request cost, not bandwidth, dominates small objects**, and listing is
  prohibitively slow on object storage. Training must read large shards named
  by an index and never list or open many small files.
- **Cold reads are limited per stream and scale with concurrency across
  objects** (22× for 32 objects). Readers need many shard downloads in flight.
- **On object storage, parallelism across objects beats byte ranges of one
  object**, and ranges cap cached reads at a fraction of a sequential read
  because each range pays an open and seek. Large payloads that must read back
  fast (checkpoints) are split into parts; range splitting is enabled only
  where the probe shows it helps both cold and cached reads (the parallel
  filesystem).
- **Cached reads are 20–60× faster than cold ones** (about 1 GB/s against
  16 MB/s per stream), so the first epoch is the bottleneck. Warming the cache
  tier ahead of training, or staging each worker's shards to local memory or
  disk, pays off.
- **Concurrent writers hurt the parallel filesystem** (33 writers reach a third
  of one writer's bandwidth), while object storage absorbs them.
- **Read-back verification is unnecessary on this mount and expensive**
  (minutes for a multi-GB checkpoint read cold). Rely on manifest SHA256 checks
  at read time instead.
- **The parallel filesystem is fast when healthy but has multi-minute write
  stalls**, so checkpoint writes should not block training (async save).
- Anything that appends (loggers) or links (artifact manifests, `latest.pt`)
  cannot live on the object-storage mount.

Real training data comes in two shapes, which need different layouts:

- **Already-sharded datasets**: hundreds to thousands of 10–20 MB shard files
  plus a few sidecars (metadata JSON, parquet tables). Per-request overhead is
  tolerable, but globbing thousands of files on object storage takes minutes
  per process. They need an index, not repacking.
- **Per-record caches**: around a million files of 1–700 KB (a feature file
  plus a sub-KB JSON sidecar per record, raw inputs, per-entry reports), read
  by random access within each worker's partition. At 0.2–0.9 s per request
  they are unusable on object storage file by file. They must be packed.

A simple rule separates the two. The fraction of time lost to per-request cost
is `t_req / (t_req + size / bandwidth)`. With `t_req` of 0.2–0.9 s and about
15 MB/s cold per stream, a 70 KB file is almost entirely overhead, a 14 MB shard
about 20–50%, and a 512 MB pack a few percent.

Moving data with a general per-file tool (copy, verify, delete source, one file
at a time) showed what the data tooling must avoid:

- Compile-cache directories moved at 30–70 KB/s (about 0.5 files/s) and did not
  finish within a day.
- Moves that were interrupted left training datasets split between tiers, with
  hundreds of thousands of files present only on the destination.
- Nothing recorded the collection as a whole, so a directory could only be
  understood by listing it, which is the slowest operation on object storage.

The data tooling must therefore pack small files, move whole collections under
a manifest written last, and delete sources only after the destination
manifest is committed and verified.

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

## Store profiles
Performance facts belong to a store, not to call sites. A `StoreProfile`
records what the probe measured:

- `request_seconds`: fixed cost of one small request;
- `stream_mb_s`: uncached bandwidth of one stream;
- `streams`: concurrent transfers worth running against the store;
- `range_streams`: parallel byte-range reads worth running on one large object
  (1 when ranges do not scale);
- `part_bytes`: range and part size;
- `list_is_cheap`: whether listing is acceptable at all.

`profile_of(store)` returns the store's profile (a conservative default when a
backend declares none). Every layer takes its defaults from it: transfer
concurrency, range splitting, prefetch depth, and `min_object_bytes(profile)`,
the size below which files should be packed rather than stored as objects. The
probe emits a recommended profile, and a private package attaches it when it
registers a scheme, e.g. `LocalStore(root, publish="exclusive", profile=...)`.

## Transfer engine
`koochak/storage/transfer.py` is the only code that moves bytes between
stores. `copy_objects(source, target, items)` takes explicit items (source
key, target key, expected size and SHA256 when a manifest provides them) and:

- runs `profile.streams` items concurrently, and reads large objects as
  `range_streams` parallel byte ranges while writing them as one sequential,
  create-only `put`;
- hashes bytes in flight and, when an expected SHA256 is given, removes a
  just-written object that does not match and raises;
- skips targets that already exist with the expected size, so interrupted
  transfers resume; refuses targets that exist with a different size;
- fails fast: the first error stops new work, waits for in-flight items, and
  raises.

Archive, pull, staging, cache warming, and checkpoint replication all call it.

## Manifested collections
One manifest format describes any immutable collection, written last and
read instead of listing. Three layouts cover datasets, archives, and
checkpoints:

- **`shards`**: files stored as-is, with size, SHA256, and optional record
  counts (the dataset index above generalizes to this).
- **`packed`**: small files packed into ~256 MB–1 GB tar packs plus a file
  table mapping each path to its pack, byte offset, size, SHA256, and mode.
  Large files (above `min_object_bytes`) stay standalone objects. Packs are
  plain tar and the manifest is JSON, so archives remain usable without
  Koochak.
- **`parts`**: a checkpoint split into parts that are written and read in
  parallel.

Listing, `stat`, and existence checks on a collection come from its manifest.

As implemented (`koochak/storage/collection.py`), the manifest is small and
references a gzipped JSON-lines file table (sorted rows, gzip mtime 0, so the
same collection always has the same bytes). Each file row records path, size,
SHA256, mode, exact mtime, group, and either `(pack, offset)` or `object`.
Grouping comes from a project-supplied table (`path,group[,order]`): Koochak
does not interpret clusters, it keeps each group contiguous in one pack in the
given order. Ownership computed at training time (which depends on world size
and worker count) is therefore not baked into the layout; a worker loads each
owned group with one range read, and staging copies an owner's share.

## Data tool
`python -m koochak.data` (phase 3) moves collections:

- `archive SRC DEST --layout packed|shards`: walk a POSIX source (listing is
  cheap there), pack or index, upload in parallel, write the manifest last.
  Deterministic ordering makes reruns resume. `--dry-run` reports file counts,
  bytes, packs, and an estimated time from the target's profile.
  `--delete-source` deletes originals only after the committed manifest has
  been verified by re-reading the destination.
- `pull SRC DEST [--include GLOB]`: restore a collection or a subset (one file
  is one range read), verify, resume.
- `ls`, `verify`, `warm`, `stage`: manifest-only listing, integrity checks,
  cache warming for backends with a cache tier, and copying a worker's or
  node's share into RAM, local disk, or another store.

Per-dataset layout files choose the layout, the grouping key for packs (for
example a cluster id, so each worker's partition maps to a few packs), and
which tiny sidecars are folded into one table.

## Readers
- **Done:** `koochak/data/packed.py` streams a grouped `packed` collection.
  `PackedGroups.plan` deals whole packs to data-loading workers (reusing
  `plan_shards`, with each pack's groups as its records); a group belongs to
  the owner of the pack holding its first file. `GroupStream` fetches whole
  packs ahead in a background thread, verifies them against the manifest,
  yields each window of packs' groups in a seeded shuffle, reshuffles every
  pass, never ends, and resumes by skipping groups without reading their
  packs. Groups that spill into another pack, and standalone objects, are
  read with range requests. `PackCache` keeps recently used packs in memory
  for data cycled faster than the rest. Per-record loaders read a group's
  files by their original paths. `PackedGroups.read_files` reads chosen
  files on their own (one range per file or run of neighbours, one open per
  pack, packs in parallel) for scattered reads outside the streams, such as one small sidecar per
  record while a dataset is built.
- `PackedTree`: a node-local cache in front of single-file reads.
- `ShardedStream`: streaming reads of a `shards` collection (below).
- Staging copies a node's owned shards or packs to faster storage before
  training.

## Where each piece lives
- **Koochak (public):** everything above, site-neutral.
- **A private package:** scheme registrations with measured profiles, layout
  files for its datasets, path-mirroring policy between tiers, and wrappers
  that run transfers as jobs on compute nodes.
- **Project repositories:** loaders read through manifests instead of listing
  directories.

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

Next (phase 4):

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

## Checkpoints (phase 5)
**Done (single-file checkpoints).** `train.checkpoint_dir` (default `out_dir`)
takes a directory or a stores-file `scheme://` URI; logs stay in `out_dir`,
since they append. Every save goes through `checkpoint.publish(store, ...)`:
the `step<N>.pt` object is written create-only, then its unchanged v1
`.ready.json` manifest; re-saving a step uncommits it first; pruning deletes
manifests before checkpoints. Directory stores keep the `latest.pt` symlink,
write-once stores get none. `train.checkpoint_async` serializes on the training
thread and publishes on one background thread (one in flight); `on_checkpoint`
fires after the manifest commits, failures surface on the next step, and
terminal, evacuation, and GPU-health saves drain first. Resume reads through
the store (so `read_settle_seconds` applies) and raises I/O errors other than a
missing file instead of falling back to an older checkpoint. The serialized
bytes are held in memory during the upload, so no node-local disk is needed.

**Remaining:** a checkpoint becomes parts plus a manifest written last:

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

## Artifacts (phase 6)
`storage.artifact.publish_artifact` uses hard links, which object-storage
mounts reject. Move ready manifests onto `Store.put`, and let the published
`path` become a store URI. Artifact gates that validate manifests must then
read through a `Store` too, which needs a coordinated protocol change in the
scheduler that consumes them.

## Phases
1. Write-once `Store`, `LocalStore`, pluggable schemes, shard index, writer
   and plan, storage probe. **Done.**
2. Store profiles, the transfer engine, and probe measurements of cold versus
   cached concurrency and of range-parallel reads of one object. **Done.**
3. Manifested collections and the data tool. **`archive` (with `--groups`),
   `pull`, `verify`, `ls`, `--files-from`, and verified moves
   (`--delete-source`) done**; `warm` and `stage` remain.
4. Readers: `PackedTree`, `ShardedStream`, staging.
5. Checkpoints through `Store`: URIs for `checkpoint_dir`, manifest-last
   publication, background saves, settling reads on resume. **Done**; parts,
   per-rank state, and replication between tiers remain.
6. Artifact manifests through `Store.put`.

## Testing Plan
- Store: round trips, byte ranges, create-only conflicts, streamed puts,
  cleanup of partial objects, prefix listing that hides in-progress writes,
  key validation, hard-link fallback error, read-back settling, URI
  resolution, registration, and entry-point discovery.
- Shards: tar determinism and compatibility with `tarfile`, writer rollover,
  commit-last, rerun idempotence, tamper detection, verification, and plan
  completeness, balance, determinism, and starvation errors.
- Probe: local run with cleanup, recommendation without hard links, CLI JSON,
  checkpoint and multi-process dataset phases, recommended profile.
- Transfer: parallel copies, range-parallel reads, resume by skipping complete
  targets, refusal of mismatched targets, SHA256 mismatch cleanup, fail-fast.
- Later phases add resume-determinism tests across partial runs and
  mocked-mount tests for exclusive publish and delayed read-back.

## Risks and Open Questions
- Read-back verification doubles checkpoint I/O on mounts that need it. None
  was needed on the probed mount, where first reads always matched.
- Measurements so far come from CPU nodes; accelerator nodes may have a
  different network path. Cold multi-stream reads of data uploaded by other
  tools (rather than written through the mount) are not yet measured.
- Whether warming the cache tier from one node makes the data cached for
  every node (a shared cache) or only for that node decides how `warm` and
  staging are scheduled.
- Cold concurrency was measured up to 32 streams from one node; the per-node
  and aggregate ceilings are not yet known.
- Exact resume across a changed world size is not possible with rank-based
  plans; the default is to fail loudly.
- Whether node-local disks exist on the target compute nodes decides whether
  staging and tiered checkpoints are worth building.
