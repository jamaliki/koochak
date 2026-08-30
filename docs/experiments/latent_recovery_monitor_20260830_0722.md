# Latent recovery monitor: 2026-08-30 07:22 UTC

No new foldability endpoint landed during this interval. The corrected
filtered ESMFold replay remains at 18 successful panels, 68 queued panels,
72 `summary.csv` files, and 1,724 actual PDBs. The existing corrected
foldability chart therefore remains the current chart; it was not regenerated
without new measurements.

The apparent lack of latent training execution is scheduling, not a dead
controller. Scruffy's state file is advancing and its only running job is the
unrelated four-GPU experiment
`gnn-sequence-residual-dropout-scratch-4gpu`. It is making real progress: the
on-disk log reaches step 282,760 and the output directory contains a verified
step 282,000 checkpoint. It occupies all four GPUs on gpu-9, so the latent
replacement jobs are queued behind it. This job was not cancelled because it
is active and unrelated to the latent campaign.

Active latent workflow state remains:

| Workflow | Succeeded | Queued | Blocked | Failed |
| --- | ---: | ---: | ---: | ---: |
| selected-4 L256 b256 recovery | 4 | 8 | 90 | 0 |
| selected-4 L256 b256 scratch | 1 | 2 | 47 | 0 |
| filtered all-16 ESMFold v2 | 18 | 68 | 0 | 0 |
| sample-analysis recovery v2 | 4 | 0 | 0 | 0 |
| filtered analysis recovery v3 | 0 | 1 | 0 | 0 |
| filtered all-atom recovery v2 | 0 | 0 | 8 | 0 |

GPU quarantine remains 0. Slurm allocation `289717` is running on
`gpu-[2,4,9,11]`; allocation `289718` is dependency-pending. No relaunch was
warranted this interval: all failures in the recent audit belong to the
superseded malformed ESMFold replay or abandoned b320 OOM runs, while active
b256 replacements have no failures.

