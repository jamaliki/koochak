# Delayed-sidechain offset best-eight length-256 campaign

## Question

Do the strongest compact offset-clock settings from the length-128 screen
retain backbone quality, sequence diversity, side-chain packing, and ESMFold
self-consistency when the training-data maximum length is raised to 256?

## Arms and fixed settings

The campaign selected a no-delay control and seven length-128 frontier arms:

| Arm | Peak ratio | Offset (A) | Special treatment |
| --- | ---: | ---: | --- |
| `ratio1_control` | 1.0 | 0.1 | no side-chain delay |
| `ratio4_offset0p05` | 4.0 | 0.05 | strong late delay |
| `ratio3_offset0p1_selfcond` | 3.0 | 0.1 | clock-gated side-chain self-conditioning |
| `ratio2_offset0p4` | 2.0 | 0.4 | earlier offset |
| `ratio3_offset0p2` | 3.0 | 0.2 | central promoted clock |
| `ratio1p5_offset0p4` | 1.5 | 0.4 | mild earlier delay |
| `ratio2_offset0p2_cbdelayed` | 2.0 | 0.2 | C-beta delayed with side chain |
| `ratio3_offset0p1_lddt` | 3.0 | 0.1 | clock-gated all-atom lDDT |

All delayed arms use onset 10 A and `alpha=4`. Training used the validated
`3/3/8/3/3` architecture, no Kabsch, polar weight 2, sequence-loss sigma ramp
0.5 to 1, and dataset filters min length 32, **max length 256**, min mean pLDDT
80, max loop length 15, max loop content 0.4, and min packing density 0.3.
Sampling was planned at 50k, 100k, 150k, and 200k for lengths 64, 128, 192,
and 256.

## Launch provenance

- Source commit: `9b7ca2f3c7e6a2f7dceeaf3b359b548e7568ab64`
- Workflow: `hk-delayed-sidechain-offset-best8-length256-9b7ca2f-v1`
- Koochak: `d186e7cc1533446165e5a92d504f2c5a0c051409`
- Root:
  `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/delayed-sidechain-offset-best8-length256-200k/9b7ca2f3c7e6a2f7dceeaf3b359b548e7568ab64/v1`

The three fixed-length throughput canaries succeeded before training admission.

## Result availability audit (2026-08-24)

**The ESMFold designability comparison was not completed.** Scruffy reports
the 50k-150k sampling and ESMFold tasks as blocked and the workflow contains no
200k ESMFold tasks. The campaign analysis directories contain launch manifests
but no milestone or campaign result JSON. A canonical-pLDDT scan of the entire
campaign root found only the single ESMFold canary prediction.

Consequently there are no valid eight-arm length-256 foldability numbers to
rank or correct. Any earlier wording implying a completed length-256
designability panel should be treated as superseded by this artifact audit.

Audit provenance:

- Source: `506c4e935cacff86dae7f649f58b4f070daa92e3`
- Workflow: `hk-esmfold-ledger-audit-length256-506c4e9-v1`
- Job: `job-26ca5555104602b7b1dc` (succeeded)
- `audit.json` SHA-256:
  `28c3d28300d0d4a0b49a7f6dc103261c8337d04d926fcbbfe489ce670d00a52e`

The corrected cross-campaign interpretation is in
[`esmfold_designability_rescore_20260824.md`](esmfold_designability_rescore_20260824.md).
