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
- AA CE is gated per example to coordinate noise `sigma <= 0.5 A`; the active
  fraction is logged so the low-noise sequence factor is auditable.
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

## Milestone sampling

Every 25k checkpoint is sampled with EMA weights and the released Pallatom
sampler: 200 perturbed-time Euler steps, recurrent previous-step coordinate
self-conditioning, gamma 0.2, noise scale 1.003, step scale 2.25, and final
temperature-0.1 softmax/argmax sequence decoding. The comparison screen uses
32 samples at each of lengths 64, 96, and 128 with the same seeds for every
variant.

```bash
python scripts/sample_short128_milestone.py \
  --config /runs/lr1e3_none/config.yaml \
  --checkpoint /runs/lr1e3_none/step000025000.pt \
  --output-dir /samples/step025000/lr1e3_none \
  --lengths 64,96,128 --samples-per-length 32 --batch-size 32 \
  --seed 20260813 --compile
```

The driver loads one checkpoint once, uses EMA unless `--raw` is explicit,
and writes a manifest with the checkpoint step, weight source, schedule, seeds,
precision, and sample counts next to the PDB/FASTA outputs.

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

At the first 25k checkpoint, runs created before resident shard ownership was
promoted can move losslessly to the current loader with:

```bash
python scripts/materialize_resident_resume.py \
  --source RUN_DIR/config.yaml \
  --output RUN_DIR/config-resident.yaml
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m hierarchical_kaveh.train \
  --config RUN_DIR/config-resident.yaml \
  --resume RUN_DIR/step000025000.pt
```

The materializer changes only `data.shard_cache_size` to `null` and sets
`wandb.resume: must`; scientific configuration and checkpoint RNG/optimizer/EMA
state remain unchanged.
