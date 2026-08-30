# Latent recovery monitor: 2026-08-30 06:22 UTC

**Scope:** selected-4 L256 continuation, scratch-from-zero control, filtered
all-16 ESMFold recovery, and their downstream replay jobs.

## Verified status

The active Scruffy queue is healthy under Slurm allocation `289717` on
`gpu-[2,4,9,11]`; successor allocation `289718` is pending on the normal
dependency. GPU quarantine is **0**, and the controller is running with the
health probe disabled. The earlier `cuda_probe_failed` quarantines were false
positives observed while GPUs were occupied; there is still no corroborating
ECC, thermal, or bad-sample evidence.

Scruffy state was audited directly from
`/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105/state.json`:

| Workflow | Succeeded | Queued | Blocked | Failed |
| --- | ---: | ---: | ---: | ---: |
| `hk-latent-selected4-length256-500k-b256-recovery` | 4 | 8 | 90 | 0 |
| `hk-latent-r1_z4_b0p03_c0p25-length256-500k-b256-scratch-v2` | 1 | 2 | 47 | 0 |
| `hk-latent-filtered-all16-400k-b81485f-recovery-esmfold-v2` | 18 | 68 | 0 | 0 |
| `hk-latent-sample-analysis-recovery-v2` | 4 | 0 | 0 | 0 |
| `hk-latent-filtered-all16-analysis-recovery-v3` | 0 | 1 | 0 | 0 |
| `hk-latent-filtered-all16-400k-b81485f-recovery-all-atom-v2` | 0 | 0 | 8 | 0 |

## On-disk output verification

Training success states correspond to actual checkpoint files, not launch
manifests:

- selected-4 recovery has four 250k checkpoints at
  `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-selected4-length256-500k-b256-recovery/b6ccee755b9420385ea09a132378e8a7af6d4193/train/step0250000/`;
- the scratch-from-zero arm has a 50k checkpoint at
  `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-r1_z4_b0p03_c0p25-length256-500k-b256-scratch-v2/b6ccee755b9420385ea09a132378e8a7af6d4193/train/step0050000/r1_z4_b0p03_c0p25/step0050000.pt`;
- the corrected filtered ESMFold replay contains 72 summaries and 1,724 PDBs
  at
  `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-filtered-all16-400k/b81485f257299928492ebc38a31e0eccb477fb02/recovery-esmfold-v2/esmfold`.

No new ESMFold panel completed since the previous ledger snapshot, so the
corrected foldability chart remains current; the next chart revision should
be made when queued panels complete. The recent failed records are confined to
the superseded v1 ESMFold replay and abandoned b320 OOM attempts; active b256
replacement workflows require no further relaunch at this audit.

