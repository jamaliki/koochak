# Old-Kaveh mechanism factorial: corrected ESMFold results

## Question

Which recovered old-Kaveh training or sampling mechanism explains the large
historical designability advantage over the delayed-sidechain models?

## Design

The 200k length-128 factorial varies four balanced training factors. Variant
bits are ordered as:

1. mixed uniform/log-uniform rather than EDM training sigma;
2. intermediate coordinate and amino-acid supervision;
3. sampled amino-acid-token recycling;
4. secondary-structure and Foldseek-3Di prediction/recycling.

Thus `f0000` is the current EDM objective without any recovered training trick.
Each sampler/cell endpoint pools 16 samples at each of lengths 64, 96, and 128
for 48 paired samples.

The four sampler presets are:

- `current`: 200 steps, rho 5, sigma 0.003 to 200, constant step scale 2.25,
  ordinary recurrent coordinate self-conditioning;
- `no_sc`: the current sampler without amino-acid or structural feedback;
- `old_struct`: 300 steps, rho 3, sigma 0.1 to 320, step scale 2.25 rising to
  4.5 on a cosine schedule, and coordinate/structural feedback only at high
  sigma (>=80) and low sigma (<=10);
- `old_full`: `old_struct` plus amino-acid self-conditioning below sigma 10 at
  temperature 0.5.

## Corrected decision

**The recovered sampler, not a recovered training trick, is the dominant
effect.** The all-zero training cell changes from 2/48 (4.2%) under `current`
sampling to 26/48 (54.2%) under `old_struct`. The paired improvement is +50.0
percentage points with a deterministic bootstrap 95% interval of +35.4 to
+64.6 points. RMSD <2 rises from 50.0% to 81.2%, corrected mean pLDDT rises
from 63.92 to 79.43, and pLDDT >80 rises from 4.2% to 54.2%.

This recovers the remembered roughly-50% independent-arm regime. It does not
support an architectural explanation: `f0000` has none of mixed-noise
training, deep supervision, AA-token recycling, or SS/3Di recycling.

## Representative 200k cells

| Training cell | Current | No SC | Old structure | Old full |
| --- | ---: | ---: | ---: | ---: |
| `f0000` | 2/48 (4.2%) | 2/48 (4.2%) | **26/48 (54.2%)** | not applicable |
| `f0010` | 13/48 (27.1%) | 5/48 (10.4%) | 20/48 (41.7%) | 18/48 (37.5%) |
| `f0100` | 8/48 (16.7%) | 4/48 (8.3%) | 15/48 (31.2%) | not applicable |
| `f0110` | 4/48 (8.3%) | 3/48 (6.2%) | 13/48 (27.1%) | 13/48 (27.1%) |
| `f0001` | 3/48 (6.2%) | 3/48 (6.2%) | 8/48 (16.7%) | not applicable |
| `f0111` | 1/48 (2.1%) | 2/48 (4.2%) | 7/48 (14.6%) | 7/48 (14.6%) |

`old_full` is only defined for cells trained with AA-token recycling. The best
AA-recycling cell is below `f0000`, so categorical AA feedback is not needed to
obtain the 54.2% result.

## Balanced 200k main effects

Effects are positive-level minus negative-level strict-designability rate,
averaged across the balanced factorial. Intervals are paired bootstraps over
matched `(length, sample_index)` rows.

| Sampler | Factor | Effect | 95% interval |
| --- | --- | ---: | ---: |
| Current | mixed sigma schedule | -8.6 points | [-11.7, -5.7] |
| Current | deep supervision | -0.8 points | [-3.6, +2.1] |
| Current | AA-token recycling | +0.8 points | [-1.3, +2.9] |
| Current | SS/3Di recycling | -4.9 points | [-8.1, -2.1] |
| Old structure | mixed sigma schedule | **-22.4 points** | [-27.1, -17.7] |
| Old structure | deep supervision | **-6.3 points** | [-10.2, -2.3] |
| Old structure | AA-token recycling | -2.1 points | [-6.3, +2.1] |
| Old structure | SS/3Di recycling | **-15.6 points** | [-19.8, -11.7] |
| Old full | mixed sigma schedule | **-19.3 points** | [-26.0, -13.0] |
| Old full | deep supervision | -0.5 points | [-5.2, +3.6] |
| Old full | SS/3Di recycling | **-12.0 points** | [-18.8, -5.7] |

The factor interactions and all 63 sampler/training cells are retained in the
machine-readable rescore artifact. Training-factor estimates remain one-seed
factorial effects; they are not training-seed confidence intervals.

## Mechanistic interpretation

The promoted next comparison should isolate the old sampler bundle rather than
retrain more architecture variants. Its changes are coupled:

- a longer 300-step integration;
- rho 3 rather than rho 5;
- sigma endpoints 0.1 and 320 rather than 0.003 and 200;
- a cosine step-scale ramp from 2.25 to 4.5 rather than constant 2.25; and
- recurrent coordinate/structural feedback gated out of the middle-noise
  interval, rather than continuously applied.

Churn is retained at gamma 0.2 in both presets, so this result does not argue
for removing churn. It says that the large missing effect lies somewhere in
the coupled time-grid, late step-scale, and recurrence-window bundle. A paired
sampler ablation of those components can now be hypothesis-driven.

## Correction provenance

The old wrapper gave `f0000`/`old_struct` only 6/48 because it averaged absent
Atom37 slots. The corrected score is 26/48. Source PDBs and CA RMSD are
unchanged.

The canonical audit used source `905d7820c3c61257e86c0a43bc03ce72be64493b`,
workflow `hk-esmfold-ledger-audit-lustre-905d782-v1`, and job
`job-c43ae6ba83aa5cf651ba` (succeeded, zero panel errors). Audit hashes:

- `audit.json`: `36e271af05769d1222cfb4ac6a0cc58991527008b2b0cdc9d4f7c67919273907`
- `rows.jsonl`: `1703efeb541a5ab62deaaaede4be06904ec740df84f2cd8b0a00ca98e6c09cf6`

The cross-campaign interpretation is in
[`esmfold_designability_rescore_20260824.md`](esmfold_designability_rescore_20260824.md).
