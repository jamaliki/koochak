# Foldability snapshot: checkpoints landed in the 12-hour window

**Recorded:** 2026-08-29 09:00 UTC snapshot  
**Scope:** verified downstream artifacts whose ESMFold panel completed during approximately 2026-08-28 21:00 to 2026-08-29 09:00 UTC.  
**Primary question:** what foldability numbers landed for the mature filtered/recycling runs and the continuing selected-4 runs?

## Important separation

The two campaigns below must not be pooled:

1. **Filtered recycling-best2** is the filtered-data campaign using the two best arms and recycling. It has both ESMFold and all-atom summaries for 100k, 150k, 200k, and 250k.
2. **Selected-4 500k continuation** is the earlier, **unfiltered** selected-4 continuation. It has ESMFold panels for 250k, 300k, 350k, and 400k, but no all-atom summaries yet. Its arm names overlap with the filtered campaign, but the dataset treatment is different.

The filtered all-16 400k campaign has real training checkpoints through 300k, but no verified sample, ESMFold, or all-atom output in this window. Its foldability is therefore not reported as zero.

## Definitions and denominators

Strict designability is:

```text
CA Kabsch RMSD < 2 A AND mean ESMFold pLDDT > 80
```

Each panel contains 32 samples per `(arm, length)` cell. Recycling checkpoints therefore contain `2 arms x 3 lengths x 32 = 192` ESMFold predictions. Selected-4 checkpoints contain `4 arms x 3 lengths x 32 = 384` predictions when all four panels are present; the 250k selected-4 snapshot is incomplete because `r0_z3_b0p3_c0p125` has no ESMFold panel.

ESMFold pLDDT is averaged over actual ATOM records in the predicted PDB. All-atom values are matched physical Atom14 heavy-atom RMSDs after CA alignment with local terminal-symmetry corrections for ASP, GLU, PHE, and TYR. These are self-consistency surrogates, not experimental folding measurements.

## Verified artifacts

| Campaign | checkpoint(s) with completed ESMFold in window | arms | ESMFold predictions | all-atom summary |
|---|---:|---|---:|---|
| filtered recycling-best2, `constant_k3` | 100k, 150k, 200k, 250k | `r1_z4_b0p03_c0p25`, `r2_z2_b0p03_c0` | 192/checkpoint | present for all four |
| selected-4 500k continuation, `constant_k3` | 250k, 300k, 350k, 400k | `r1_z3_b0p01_c0p125`, `r1_z4_b0p03_c0p25`, `r0_z3_b0p3_c0p125`, `r2_z2_b0p1_c0p125` | 288 at 250k; 384 at 300-400k | absent |

Remote roots:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-filtered-recycling-best2-400k/a5728e6d5b578b4f61df409a09faae642c521180
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-selected4-500k/49b037f10a9916b6f8ad8bfc1da07928ff47411f
```

## Aggregate foldability: filtered recycling-best2

The aggregate is across both arms and all three lengths, with `n=192` per checkpoint.

| step | designable / 192 | designability | mean pLDDT | mean CA RMSD (A) | mean all-atom RMSD (A) |
|---:|---:|---:|---:|---:|---:|
| 100k | 74 | 38.54% | 75.23 | 2.33 | 2.82 |
| 150k | 56 | 29.17% | 70.03 | 2.96 | 3.50 |
| 200k | 44 | 22.92% | 66.11 | 3.96 | 4.56 |
| 250k | 43 | 22.40% | 64.71 | 3.75 | 4.38 |

The all-atom aggregate is the mean of the two arm-level all-atom summaries; it is reported as an RMSD summary, not as a second designability endpoint.

### Recycling, disaggregated by arm and length

| step | arm | length | designable / 32 | designability | mean pLDDT | CA RMSD (A) | all-atom RMSD (A) |
|---:|---|---:|---:|---:|---:|---:|---:|
|100k|r1_z4_b0p03_c0p25|64|20|62.50%|80.10|1.40|1.90|
|100k|r1_z4_b0p03_c0p25|96|6|18.75%|71.59|1.80|2.18|
|100k|r1_z4_b0p03_c0p25|128|14|43.75%|78.51|2.09|2.47|
|100k|r2_z2_b0p03_c0|64|14|43.75%|76.58|2.48|3.04|
|100k|r2_z2_b0p03_c0|96|9|28.13%|68.62|3.19|3.70|
|100k|r2_z2_b0p03_c0|128|11|34.38%|75.99|3.05|3.62|
|150k|r1_z4_b0p03_c0p25|64|23|71.88%|81.01|1.44|2.03|
|150k|r1_z4_b0p03_c0p25|96|2|6.25%|62.92|3.10|3.64|
|150k|r1_z4_b0p03_c0p25|128|7|21.88%|69.71|2.89|3.31|
|150k|r2_z2_b0p03_c0|64|17|53.13%|79.52|1.99|2.61|
|150k|r2_z2_b0p03_c0|96|1|3.13%|60.84|4.68|5.21|
|150k|r2_z2_b0p03_c0|128|6|18.75%|66.17|3.63|4.21|
|200k|r1_z4_b0p03_c0p25|64|25|78.13%|81.81|1.23|1.76|
|200k|r1_z4_b0p03_c0p25|96|0|0.00%|56.46|5.37|5.94|
|200k|r1_z4_b0p03_c0p25|128|4|12.50%|64.70|4.03|4.57|
|200k|r2_z2_b0p03_c0|64|9|28.13%|72.80|2.46|3.18|
|200k|r2_z2_b0p03_c0|96|3|9.38%|59.27|5.20|5.74|
|200k|r2_z2_b0p03_c0|128|3|9.38%|61.62|5.46|6.15|
|250k|r1_z4_b0p03_c0p25|64|21|65.63%|81.40|1.03|1.61|
|250k|r1_z4_b0p03_c0p25|96|2|6.25%|57.10|5.73|6.26|
|250k|r1_z4_b0p03_c0p25|128|3|9.38%|57.20|5.18|5.73|
|250k|r2_z2_b0p03_c0|64|13|40.63%|73.98|1.87|2.47|
|250k|r2_z2_b0p03_c0|96|2|6.25%|59.70|3.80|4.44|
|250k|r2_z2_b0p03_c0|128|2|6.25%|58.85|4.92|5.65|

Arm-level all-atom aggregates (96 samples/arm/checkpoint) were:

| step | arm | designable / 96 | designability | mean pLDDT | CA RMSD (A) | all-atom RMSD (A) |
|---:|---|---:|---:|---:|---:|---:|
|100k|r1_z4_b0p03_c0p25|40|41.67%|76.73|1.76|2.18|
|100k|r2_z2_b0p03_c0|34|35.42%|73.73|2.91|3.46|
|150k|r1_z4_b0p03_c0p25|32|33.33%|71.22|2.48|2.99|
|150k|r2_z2_b0p03_c0|24|25.00%|68.84|3.43|4.01|
|200k|r1_z4_b0p03_c0p25|29|30.21%|67.66|3.54|4.09|
|200k|r2_z2_b0p03_c0|15|15.63%|64.56|4.37|5.02|
|250k|r1_z4_b0p03_c0p25|26|27.08%|65.23|3.98|4.53|
|250k|r2_z2_b0p03_c0|17|17.71%|64.18|3.53|4.19|

## Aggregate foldability: selected-4 500k continuation

These are ESMFold-only values. At 250k, the `r0_z3_b0p3_c0p125` panel was missing, so the aggregate is over three arms (`n=288`); at 300k-400k it is over four arms (`n=384`).

| step | arms present | designable / n | designability | mean pLDDT | mean CA RMSD (A) |
|---:|---:|---:|---:|---:|---:|
|250k|3|117 / 288|40.63%|75.92|2.26|
|300k|4|171 / 384|44.53%|77.74|2.17|
|350k|4|208 / 384|54.17%|79.31|2.00|
|400k|4|195 / 384|50.78%|78.67|1.75|

### Selected-4, disaggregated by arm and length

| step | arm | L64 | L96 | L128 | mean designability | mean pLDDT | mean CA RMSD (A) |
|---:|---|---:|---:|---:|---:|---:|---:|
|250k|r1_z3_b0p01_c0p125|25/32 (78.13%)|13/32 (40.63%)|8/32 (25.00%)|46/96 (47.92%)|77.71|1.66|
|250k|r1_z4_b0p03_c0p25|21/32 (65.63%)|16/32 (50.00%)|5/32 (15.63%)|42/96 (43.75%)|78.84|2.05|
|250k|r2_z2_b0p1_c0p125|18/32 (56.25%)|8/32 (25.00%)|3/32 (9.38%)|29/96 (30.21%)|71.20|3.09|
|300k|r0_z3_b0p3_c0p125|25/32 (78.13%)|14/32 (43.75%)|9/32 (28.13%)|48/96 (50.00%)|79.06|1.53|
|300k|r1_z3_b0p01_c0p125|24/32 (75.00%)|15/32 (46.88%)|7/32 (21.88%)|46/96 (47.92%)|78.07|2.67|
|300k|r1_z4_b0p03_c0p25|22/32 (68.75%)|15/32 (46.88%)|13/32 (40.63%)|50/96 (52.08%)|80.61|1.62|
|300k|r2_z2_b0p1_c0p125|19/32 (59.38%)|5/32 (15.63%)|3/32 (9.38%)|27/96 (28.13%)|73.23|2.87|
|350k|r0_z3_b0p3_c0p125|26/32 (81.25%)|17/32 (53.13%)|8/32 (25.00%)|51/96 (53.13%)|79.64|1.50|
|350k|r1_z3_b0p01_c0p125|29/32 (90.63%)|17/32 (53.13%)|6/32 (18.75%)|52/96 (54.17%)|80.00|1.58|
|350k|r1_z4_b0p03_c0p25|26/32 (81.25%)|21/32 (65.63%)|13/32 (40.63%)|60/96 (62.50%)|81.12|1.65|
|350k|r2_z2_b0p1_c0p125|28/32 (87.50%)|13/32 (40.63%)|4/32 (12.50%)|45/96 (46.88%)|76.37|3.02|
|400k|r0_z3_b0p3_c0p125|22/32 (68.75%)|20/32 (62.50%)|10/32 (31.25%)|52/96 (54.17%)|80.00|1.58|
|400k|r1_z3_b0p01_c0p125|27/32 (84.38%)|17/32 (53.13%)|18/32 (56.25%)|62/96 (64.58%)|81.12|1.31|
|400k|r1_z4_b0p03_c0p25|24/32 (75.00%)|11/32 (34.38%)|6/32 (18.75%)|41/96 (42.71%)|77.20|1.96|
|400k|r2_z2_b0p1_c0p125|21/32 (65.63%)|15/32 (46.88%)|4/32 (12.50%)|40/96 (41.67%)|76.36|2.16|

For the selected-4 arm panels, mean residue-indexed TM scores ranged from 0.78 to 0.89 at the checkpoint aggregates. The full per-cell TM and ESMFold fields remain in the verified remote panel summaries; the companion JSON here contains the aggregate plotting inputs.

## What the window teaches us

* The **filtered recycling** campaign is not improving with additional steps in this interval. The best aggregate checkpoint is 100k (38.54%), followed by 150k (29.17%), 200k (22.92%), and 250k (22.40%). The collapse is driven primarily by lengths 96 and 128: by 200k, the `r1_z4` 96-residue cell is 0/32 and its 128-residue cell is 4/32.
* The **unfiltered selected-4 continuation** is materially stronger at maturity than the recycling campaign: its best arm/checkpoint is `r1_z3_b0p01_c0p125` at 400k, 62/96 = 64.58%. The best 400k aggregate is 50.78% across all four arms. The 350k `r1_z4` arm is also strong at 60/96 = 62.50%.
* The selected-4 panel is not monotonic by arm. `r1_z3` improves from 47.92% at 250k to 64.58% at 400k, while `r1_z4` peaks at 62.50% at 350k and falls to 42.71% at 400k. `r2_z2` remains weakest at longer lengths.
* The direct comparison is therefore not “more steps always helps”: recycling appears to destabilize the longer-length distribution, whereas the selected-4 continuation retains or improves foldability through 400k. This is a campaign/dataset/architecture comparison, not a controlled checkpoint comparison, because the recycling run is filtered and the selected-4 run is unfiltered.

## Incomplete or repaired downstream work

* Filtered all-16 campaign: checkpoints through 300k are real, but no verified downstream samples/ESMFold/all-atom summaries were present at snapshot time. A resubmission of workflow `hk-latent-filtered-all16-400k-b81485f-v1` was attempted after a dry-run passed; the client deadline expired after launch. Reconciliation found 14 durable preflight reports but no new downstream outputs, so the uncertain mutation was not replayed.
* Selected-4 250k: ESMFold is missing only for `r0_z3_b0p3_c0p125`; 300k-400k have all four ESMFold panels. All-atom postprocessing is still absent for this campaign.
* Filtered recycling 300k: training checkpoints exist, but sampling and downstream analysis have not yet landed. The existing recycling workflow remains the correct source for deduplicated continuation.

## Machine-readable companion

[Foldability snapshot data and chart inputs](artifacts/filtered_checkpoint_foldability_12h.json)  
[Checkpoint trajectory chart](artifacts/filtered_checkpoint_foldability_12h.svg)
