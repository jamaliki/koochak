# New latent sample analyses: 2026-08-30

**Status:** Complete for the newly durable panels listed below  
**Analysis output:** `final.json` from the actual sample-analysis jobs  
**Metric scope:** `constant_k3` variant; geometry, latent usage, and sequence composition

## Panels and provenance

| Panel | Samples | Code commit / campaign |
| --- | ---: | --- |
| filtered `r0_z3_b0p03_c0p5` at 350k | 96 | `b81485f257299928492ebc38a31e0eccb477fb02`, filtered all-16 recovery v3 |
| selected-16 `r1_large_z4_b0p01_c0p25` at 150k | 96 | `49b037f10a9916b6f8ad8bfc1da07928ff47411f`, downstream recovery v4 |
| selected-16 `r1_large_z8_b0p01_c0p125` at 150k | 96 | same |
| selected-16 `r0_large_z8_b0p01_c0p125` at 150k | 96 | same |
| selected-4 `r0_z3_b0p3_c0p125` at 250k | 224 | `b6ccee755b9420385ea09a132378e8a7af6d4193`, L256 b256 recovery |
| selected-4 `r1_z3_b0p01_c0p125` at 250k | 224 | same |
| selected-4 `r1_z4_b0p03_c0p25` at 250k | 224 | same |
| selected-4 `r2_z2_b0p1_c0p125` at 250k | 224 | same |
| scratch `r1_z4_b0p03_c0p25` at 50k | 224 | `b6ccee755b9420385ea09a132378e8a7af6d4193`, L256 b256 scratch-v2 |

The complete machine-readable table is
[`artifacts/latent_analysis_new_panels_20260830.csv`](artifacts/latent_analysis_new_panels_20260830.csv), with the matched chart at
[`artifacts/latent_analysis_new_panels_20260830.png`](artifacts/latent_analysis_new_panels_20260830.png).

Remote `final.json` roots:

- filtered: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-filtered-all16-400k/b81485f257299928492ebc38a31e0eccb477fb02/recovery-analysis-v3/step0350000/r0_z3_b0p03_c0p5/final.json`;
- selected-16: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-selected16-200k/49b037f10a9916b6f8ad8bfc1da07928ff47411f/recovery-v4/sample-analysis-recovery-full/step0150000/`;
- selected-4: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-selected4-length256-500k-b256-recovery/b6ccee755b9420385ea09a132378e8a7af6d4193/sample-analysis/step0250000/`;
- scratch: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-r1_z4_b0p03_c0p25-length256-500k-b256-scratch-v2/b6ccee755b9420385ea09a132378e8a7af6d4193/sample-analysis/step0050000/`.

## Findings

### 1. Geometry is broadly sane across all nine panels

The CA step-bad/cis-peptide fractions are at most `4.93e-4`, nonlocal CA
clash fractions are zero to `6.34e-6`, and normalized radius of gyration is
`2.685-2.963`. These analyses do not show a new steric or chain-geometry
failure. The selected-4 250k panels are somewhat more repetitive in sequence,
but not geometrically broken.

### 2. The strongest selected-4 arm is still `r1_z4_b0p03_c0p25`

At 250k it has effective latent rank **3.65/4**, all four dimensions active,
the highest latent/BLOSUM correspondence among the selected-4 panels
(**0.582**), entropy **3.046 bits**, effective alphabet **8.62**, and maximum
residue fraction **0.273**. The scratch-from-zero panel for the same arm at
50k is already similar: rank **3.70/4**, correspondence **0.581**, entropy
**2.998 bits**, alphabet **8.29**, and maximum residue fraction **0.281**.
This is an encouraging early scratch result, but it is not a foldability result
and it does not establish that scratch training will match the resumed run at
later checkpoints.

### 3. The other selected-4 arms show latent/compositional collapse

`r2_z2_b0p1_c0p125` is the weakest: rank **1.96/4**, only two active
dimensions, correspondence **0.108**, entropy **2.259 bits**, alphabet **5.64**,
and maximum residue fraction **0.444**. `r0_z3_b0p3_c0p125` and
`r1_z3_b0p01_c0p125` are intermediate but similarly concentrated, with
effective alphabets **6.94** and **7.76** and maximum residue fractions
**0.368** and **0.383**. Their unique-sequence fraction is still 1.0, so this
is compositional bias rather than duplicate sampled strings.

### 4. The selected-16 z8 arms use their latent space much better

Both z8 panels at 150k activate all eight dimensions and have effective rank
about **7.79**, compared with **3.80/4** for the matched z4 panel. They also
have higher sequence entropy (**3.60-3.64 bits**) and broader effective
alphabets (**12.19-12.54**) than the z4 panel (**3.08 bits**, **8.67**).
The two z8 arms are very similar in these diagnostics; the `r0` z8 arm has
slightly lower maximum residue fraction (0.172 versus 0.192) and slightly
higher entropy.

### 5. The filtered 350k panel is compositionally healthier, but not directly
comparable as a causal filter test

The filtered `r0_z3_b0p03_c0p5` panel has entropy **3.867 bits**, effective
alphabet **14.65**, and maximum residue fraction **0.142**, much healthier
than the selected-4 `r0_z3` panel. Its latent rank is **2.99/3** with all three
dimensions active. However, it changes both the arm and the data/training
campaign, so this is evidence of a healthier sampled distribution, not a
matched estimate of the filtering effect.

## What this does and does not establish

These are sample-analysis diagnostics, not designability or ESMFold confidence
measurements. They establish that the new analysis jobs are producing valid,
non-duplicated samples and expose clear differences in latent utilization and
sequence concentration. The next decisive endpoint is the matched ESMFold
panel for selected-4 250k and scratch 50k, followed by the same readout at
each 50k checkpoint. In particular, `r1_z4_b0p03_c0p25` remains the best
selected-4 candidate to watch, while `r2_z2_b0p1_c0p125` should be treated as
a collapse-risk arm unless its later checkpoints recover.

