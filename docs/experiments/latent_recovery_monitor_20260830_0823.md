# Latent recovery monitor: 2026-08-30 08:23 UTC

No new latent training, sampling, ESMFold, or analysis endpoint completed in
this interval. The corrected filtered ESMFold recovery remains at 18
successful panels, 68 queued, 72 summaries, and 1,724 actual PDBs; its
existing chart remains current.

The apparent latent-training stall is still scheduling. The only running
Scruffy job is the unrelated four-GPU
`gnn-sequence-residual-dropout-scratch-4gpu` run on gpu-9. Its real training
log advanced to step 282,870 at 08:23 UTC, so it was not cancelled or
classified as wedged. It occupies the four-GPU reservation needed by the
latent queue.

The active latent recovery workflows remain failure-free: selected-4 b256
recovery `4 succeeded / 8 queued / 90 blocked`, scratch b256 `1 / 2 / 47`,
filtered ESMFold v2 `18 / 68 / 0`, sample-analysis recovery v2 `4 / 0 / 0`,
filtered analysis recovery v3 `0 / 1 / 0`, and all-atom recovery v2 `0 / 0 / 8`.
GPU quarantine remains 0. No relaunch was warranted.

