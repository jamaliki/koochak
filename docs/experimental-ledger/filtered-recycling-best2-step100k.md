# Filtered latent recycling-best2: 100k result

## Decision summary

This is the first verified downstream result from the filtered recycling-best2
continuation. Both selected arms reached a real 100,000-step checkpoint and
completed sampling, sample analysis, ESMFold, and decoded all-atom RMSD.

The z4 arm, `r1_z4_b0p03_c0p25`, is the stronger arm at this checkpoint:
**40/96 designable (41.7%)**, compared with **34/96 (35.4%)** for
`r2_z2_b0p03_c0`. The z4 arm is better on mean CA RMSD, backbone RMSD, and
all-atom RMSD. This is a single-seed, checkpoint result, not yet evidence that
recycling or filtering causes the improvement.

The result is length-dependent but not monotonic in the problematic direction
seen in some earlier panels. The z4 arm has 62.5% designability at L64, 18.8%
at L96, and 43.8% at L128. The z2 arm has 43.8%, 28.1%, and 34.4%,
respectively. These estimates are based on 32 samples per length and should be
treated as descriptive.

## Exact experimental contract

| Item | Value |
| --- | --- |
| Campaign | `atom4-latent-filtered-recycling-best2-400k` |
| Source commit | `a5728e6d5b578b4f61df409a09faae642c521180` |
| Checkpoint | 100,000 steps |
| Arms | `r1_z4_b0p03_c0p25`, `r2_z2_b0p03_c0` |
| Filter | loop content <0.5; max loop length <15; packing density >0.3 |
| Sampling | constant-k3 schedule; L64/L96/L128; 32 samples per length |
| Strict endpoint | CA RMSD <2 A and mean ESMFold pLDDT >80 |
| Remote root | `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-filtered-recycling-best2-400k/a5728e6d5b578b4f61df409a09faae642c521180` |

The machine-readable values are in
[filtered_recycling_best2_step100k.json](artifacts/filtered_recycling_best2_step100k.json).
The chart is [filtered_recycling_best2_step100k.svg](artifacts/filtered_recycling_best2_step100k.svg)
and a raster copy is available at
[filtered_recycling_best2_step100k.png](artifacts/filtered_recycling_best2_step100k.png).

## Results by arm and length

### Strict designability and decoded all-atom quality

| Arm | L | Designable | Designability | Mean pLDDT | CA RMSD (A) | All-atom RMSD, symmetry-corrected (A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r1_z4_b0p03_c0p25` | 64 | 20/32 | 62.5% | 80.10 | 1.398 | 1.898 |
| `r1_z4_b0p03_c0p25` | 96 | 6/32 | 18.8% | 71.59 | 1.799 | 2.176 |
| `r1_z4_b0p03_c0p25` | 128 | 14/32 | 43.8% | 78.51 | 2.085 | 2.468 |
| `r2_z2_b0p03_c0` | 64 | 14/32 | 43.8% | 76.58 | 2.479 | 3.043 |
| `r2_z2_b0p03_c0` | 96 | 9/32 | 28.1% | 68.62 | 3.190 | 3.704 |
| `r2_z2_b0p03_c0` | 128 | 11/32 | 34.4% | 75.99 | 3.055 | 3.618 |
| **`r1_z4_b0p03_c0p25` aggregate** | - | **40/96** | **41.7%** | **76.73** | **1.761** | **2.181** |
| **`r2_z2_b0p03_c0` aggregate** | - | **34/96** | **35.4%** | **73.73** | **2.908** | **3.455** |

The ESMFold panel contains 192 predictions total, 96 per arm. Overall, 74/192
(38.5%) pass the strict endpoint; mean pLDDT is 75.23 and mean CA RMSD is
2.334 A. All-atom evaluation reproduced the CA values with a maximum absolute
regression error of 0.000093 A. No separate all-atom pass threshold was
introduced.

## Interpretation

- The z4 latent arm is the current filtered-recycling leader at 100k.
- Its advantage is broad: it is better at every tested length on CA RMSD and
  all-atom RMSD, while designability is higher at L64 and L128 and lower only
  in the relative gap at L96.
- The z2 arm has only two active latent dimensions in its sample analysis,
  whereas the z4 arm uses all four dimensions with effective rank 3.89. This
  is consistent with, but does not prove, a latent-capacity contribution to
  the quality gap.
- The L96 dip in the z4 arm and the small sample size mean that the apparent
  arm ranking needs confirmation at later checkpoints and with matched
  replicates.

## Campaign maturity and failures

The 100k arm-level records are complete:

- training: numbered `step0100000.pt` checkpoints for both arms;
- sampling: panel manifests and generated panels for both arms;
- sample analysis: `final.json` for both arms;
- ESMFold: 192 successful predictions and per-cell summaries;
- all-atom RMSD: 192 rows, summary, and report.

The 150k-400k directories currently contain launch manifests but no verified
training checkpoints for this campaign. They must not be reported as completed
results. The separate filtered all-16 400k campaign is also manifest-only at
the inspected 400k milestone; it has no verified checkpoint, sample panel,
ESMFold summary, or all-atom summary. No new repair submission was made in
this pass because there was no active latent job and no safely identifiable
failed downstream task with a durable producer checkpoint.

## Next decision

Continue the two-arm filtered recycling run to 200k before promoting it, and
repeat the same fixed panel at 100k/200k with at least one matched replicate.
Prioritize the z4 arm for the first full 400k downstream panel, but retain the
z2 arm as a capacity control. Do not infer that filtering is beneficial from
this comparison alone; the correct comparison is against the same arms,
sampler, seed, and checkpoint without filtering.
