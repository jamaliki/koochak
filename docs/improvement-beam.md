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

## Decision state

No promoted winner exists before the 100k screen. All eight runs use identical
data eligibility, seed, batch shape, self-conditioning schedule, architecture,
and runtime. Negative or neutral main effects remain evidence; they are not
silently discarded.

