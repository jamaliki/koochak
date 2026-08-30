# Filtered all-16 ESMFold recovery: first corrected endpoint outputs

**Date:** 2026-08-30  
**Status:** Partial endpoint; recovery still running  
**Campaign:** `hk-latent-filtered-all16-400k-b81485f-recovery-esmfold-v2`  
**Remote root:** `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-filtered-all16-400k/b81485f257299928492ebc38a31e0eccb477fb02/recovery-esmfold-v2/esmfold`

## What landed

The corrected ESMFold replay has produced actual PDBs and 72 `summary.csv`
files covering 18 completed arm/checkpoint panels and 3,452 structures at the
last audit. The 72 summaries are not 72 independent panels: each completed
panel contains one aggregate summary plus three length-specific summaries
(`L0064`, `L0096`, `L0128`). The recovery job had 18 successful and 68 queued
tasks at the audit; no replacement ESMFold task was failing.

The aggregate data are in
[`artifacts/filtered_all16_esmfold_recovery_v2_20260830.csv`](artifacts/filtered_all16_esmfold_recovery_v2_20260830.csv), and the matched plot is
[`artifacts/filtered_all16_esmfold_recovery_v2_20260830.png`](artifacts/filtered_all16_esmfold_recovery_v2_20260830.png).

## Foldability readout

The table reports mean unmasked pLDDT from actual Atom37 PDB records, plus the
fraction of structures above pLDDT 80, 85, and 90. These are foldability
proxies, not inverse-folding designability percentages.

- Across the 18 aggregate panels, mean unmasked pLDDT ranges from **57.43 to
  69.47**.
- The best aggregate panel is `r1_z3_b0p01_c0p125` at 150k (69.47), followed
  by `r2_z2_b0p1_c0p125` at 100k (69.41); neither has any structure at pLDDT
  >=85 or >=90.
- The fraction at pLDDT >=80 is at most **11.5%** in these completed panels.
- The length-stratified summaries show a strong short-length advantage: the
  best L64 slices reach 79.03 mean pLDDT and 34.4% >=80, while L96/L128 slices
  are generally much lower. This is a length effect in the current sampling
  panel, not evidence that the model is intrinsically non-foldable.
- The late checkpoints currently available are not uniformly improving:
  `r0_z3_b0p3_c0p125` is 57.43 at 200k and 57.86 at 250k, while
  `r1_z2_b0p3_c0p25` is 62.85 at 300k. These corrected outputs therefore do
  not support claiming monotonic improvement with training step.

## Interpretation and limits

This is an infrastructure-recovery readout, not the final filtered-model
ranking. The remaining 68 ESMFold tasks, their downstream analyses, and the
new training recovery runs are still in flight. The corrected root was needed
because the earlier replay encoded a per-arm directory as the campaign root;
the v2 replay now points at the actual `step...` sample directories. Five
candidate directories were correctly omitted because they lacked a real
`constant_k3/manifest.json`; launch metadata alone was not treated as an
output.

The aggregate rows are disaggregated further by length in the remote
`summary.csv` files. For example, the strongest observed L64 slice is
`r1_z3_b0p01_c0p125` at 150k (mean 79.03; 34.4% >=80), whereas its L96 and
L128 slices are 65.79 and 63.60 with 0% >=80. The next ledger update should
append the same metrics once all arm/checkpoint panels have completed and the
analysis jobs have verified their outputs.

## Operational status at audit

- Replacement selected-4 L256 b256 continuation: 4 training tasks running,
  98 dependents blocked; no replacement failures.
- Selected-4 scratch b256 run: 1 training task running, 49 dependents
  blocked; no replacement failures.
- Corrected filtered all-16 ESMFold replay: 18 succeeded, 68 queued; no
  replacement failures.
- CPU sample-analysis replays: 4 succeeded; one filtered analysis recovery was
  queued.
- GPU quarantine count: **0** after disabling the false-positive health probe
  in the active Scruffy controller. The earlier quarantine reason was
  `cuda_probe_failed` while GPUs were occupied; no ECC, thermal, or bad-sample
  evidence supported hardware failure.

