# Short-128 100k architecture screen

## Question

Does the fixed hierarchical architecture train stably at short context, and how
do smooth-lDDT supervision, residue-distogram supervision, and learning rate
change its common denoising diagnostics?

## Fixed baseline

- Atom14 local encoder / global residue encoder / p=4 coarse trunk / global
  residue decoder / Atom14 local decoder depths: `3 / 3 / 8 / 3 / 3`.
- The saved atom encoder state is injected before the atom decoder. The saved
  fine residue encoder state is the residual base used by unpatchification.
- Every one of eight coarse layers runs pair-biased attention plus outgoing and
  incoming triangle multiplication. There is no triangle attention or pair
  refresh.
- Maximum length 128, batch size 32 per H100, BF16, compilation and fused paths
  required, seed 42.
- Standard EDM coordinate loss and 20-class AA CE are always enabled.
- One deterministic batch-level self-conditioning Bernoulli with `p=0.5`;
  identical `(seed, step)` gives the same decision in every variant and after
  resume.
- Constant EMA decay 0.999, updated every two optimizer steps with compensated
  decay.
- Production metrics go to the new W&B project
  `hierarchical-kaveh-short128-100k`, group `short128-100k-3a3r8c`.
- Eligible samples satisfy the strict pre-sharding filters `mean_plddt > 80`
  and `loop_content < 0.5`; `loop_content` is the shard's coil-fraction field.

## Factorial variants

| ID | learning rate | smooth-lDDT weight | distogram weight |
|---|---:|---:|---:|
| `lr1e3_none` | 1e-3 | 0 | 0 |
| `lr1e3_lddt` | 1e-3 | 1 | 0 |
| `lr1e3_dist` | 1e-3 | 0 | 0.5 |
| `lr1e3_both` | 1e-3 | 1 | 0.5 |
| `lr3e4_none` | 3e-4 | 0 | 0 |
| `lr3e4_lddt` | 3e-4 | 1 | 0 |
| `lr3e4_dist` | 3e-4 | 0 | 0.5 |
| `lr3e4_both` | 3e-4 | 1 | 0.5 |

This complete `2 x 2 x 2` design estimates all main effects and interactions.
Total training loss is not comparable across objective variants. Comparisons
must use the shared coordinate, AA, and smooth-lDDT diagnostics, optimization
stability, fixed-checkpoint samples, and later a common checkpoint evaluation.

## Promotion and safety gates

Before the 100k jobs, every materialized config must pass a real-data H100 step
with finite component losses and gradients, fused kernels required, and a
loadable checkpoint. Long jobs retain checkpoints at 25k-step intervals.

No variant is promoted on training loss alone. A credible winner must remain
finite, avoid persistent gradient clipping or collapse, improve common metrics,
and later survive the same fixed sampling/evaluation panel. Throughput is a
guardrail rather than the primary objective in this round.

## Reproduction

Materialize immutable preflight configs with:

```bash
python scripts/materialize_short128_screen.py \
  --base-config configs/experiments/short128_100k.yaml \
  --metadata /absolute/path/to/metadata.json \
  --output-root /absolute/path/to/preflight \
  --preflight
```

Omit `--preflight` for the 100k configs. Production materialization is the only
mode that enables W&B, using project `hierarchical-kaveh-short128-100k`.
