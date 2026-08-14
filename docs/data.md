# Ragged Atom14 shard format

Training reads the existing Kaveh compressed NPZ shards. Dataset construction is
intentionally not copied into this repository.

## `metadata.json`

The root is either a list of shard entries or a mapping whose `shards` (or
`entries`) value is that list. Each entry must contain:

```json
{
  "shard": "shard_00000.npz"
}
```

`shard` may be absolute or relative to `metadata.json`. Additional historical
metadata fields are ignored.

## NPZ arrays

Required arrays are ragged-concatenated over residues:

| key | shape | meaning |
|---|---:|---|
| `sample_offsets` | `[samples+1]` | residue slice boundaries, beginning at zero |
| `pos` | `[total_residues,14,3]` | Atom14 coordinates in Angstroms |
| `mask` | `[total_residues,14]` | physical atom availability |
| `aatype` | `[total_residues]` | integer amino-acid type |
| `chain_idx` | `[total_residues]` | chain identifier |
| `res_idx` | `[total_residues]` | residue index |

The reader converts physical Atom14 to Pallatom-style unified Atom14. All 14
slots of a C-alpha-resolved residue are model tokens in both training and
sampling. Chemically nonexistent slots take the residue's C-alpha coordinate
and remain supervised virtual targets. A separate coordinate mask excludes
experimentally unresolved real atoms from coordinate and smooth-lDDT objectives;
their model input uses a C-alpha placeholder. Residues without a resolved
C-alpha are removed.
An internal unresolved gap therefore remains visible as a residue-index break.
The sample is then centered, and geometric chain breaks are inferred from
adjacent C-alpha distances above 4 Angstrom.

## Loading behavior

Samples are filtered/cropped to configured lengths, partitioned without overlap
over DDP ranks and DataLoader workers, grouped by length, then rigidly augmented
and corrupted as `x_t = x0 + sigma * epsilon`. Sigma is sampled independently
for each structure as `sigma_data * exp(N(-1.2, 1.5^2))`. Batches are emitted
as already-batched prefix-padded mappings. The
DataLoader therefore uses `batch_size=None`; Koochak must not shard it again.

Whole shards, rather than individual samples, are assigned without overlap to
global `(rank, worker)` owners and balanced by eligible sample count. With the
default `data.shard_cache_size: null`, each worker preloads its complete owned
shard set and the existing random sample pool then draws entirely from RAM.
Set a positive integer only for a memory-constrained bounded LRU. No pLDDT
bucket, ambient target, rotamer resampling, or sidechain-specific noise path
exists.

The number of shards should be at least the total number of DataLoader workers
across all ranks. Reduce `data.num_workers` if a worker reports an empty
partition.
