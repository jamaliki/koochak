# Improvement beam

## Promoted baseline

Commit prior to this campaign: `557c8646865d772c28506ef40bacdf1e58ddb5a2`.
The first training baseline is the fixed `3/3/8/3/3`, p=4 architecture in
[`experiments/short128_100k_screen.md`](experiments/short128_100k_screen.md).

## Active beams

| beam | type | hypothesis | decisive experiment | status |
|---|---|---|---|---|
| `OBJ-LDDT` | exploit | all-atom local-distance supervision improves coordinate geometry beyond aligned EDM MSE | lDDT off/on main effect at both LRs and distogram states | queued |
| `OBJ-DIST` | complementary | stable coarse-pair supervision improves global topology and pair-state learning | distogram off/on main effect at both LRs and lDDT states | queued |
| `OPT-LR` | optimization | the deeper 3/3/8/3/3 network benefits from the lower 3e-4 update scale | 1e-3 vs 3e-4 across all objective states | queued |
| `INTERACTION` | structural | local all-atom and coarse residue-pair losses are complementary rather than redundant | full three-factor interaction | queued |
| `IO-SHARD-OWNERSHIP` | systems | sample-strided workers repeatedly decompress the same 4,606 Lustre shards despite abundant host RAM | compare immutable `d68c966` against disjoint global-worker shard ownership with full owned-shard preload; require lower p95 and higher completed-step throughput without changing eligible samples | active |

## Short-128 latency tail

The initial eight-run screen exposed a systems tail rather than a model change:
after step 1,000, median logged enqueue time is roughly 0.18--0.22 s while p95
is 1.78--2.20 s. The spikes are synchronized across identically seeded runs.
The dataset contains 4,606 shards (52.8 GiB compressed); the arrays used by
training occupy about 27.2 MiB per shard median. A full decompressed copy is
therefore about 125 GiB per run, which fits comfortably in the Tokyo node RAM.

Candidate `IO-SHARD-OWNERSHIP` assigns every shard to exactly one global
`(rank, worker)` owner, balances owners by eligible sample count, preloads each
owner's full working set, and retains the existing shuffled sample pool. The
scientific sample set, corruption, and model are unchanged. Promotion requires:

- exhaustive, non-overlapping deterministic ownership at one and multiple ranks;
- identical per-sample tensors and finite training losses/gradients;
- no DataLoader queue starvation after warmup;
- at least 20% lower p95 completed-step latency and no median/throughput regression
  in paired one-H100 tests using the real filtered dataset;
- bounded resident memory well below the job allocation.

## Decision state

No promoted winner exists before the 100k screen. All eight runs use identical
data eligibility, seed, batch shape, self-conditioning schedule, architecture,
and runtime. Negative or neutral main effects remain evidence; they are not
silently discarded.
