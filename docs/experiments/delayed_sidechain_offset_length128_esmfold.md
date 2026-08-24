# Length-128 offset-clock ESMFold designability follow-up

## Question

Which delayed-sidechain offset variant produces sequences that ESMFold refolds
to the generated backbone with high confidence, and how does that result evolve
over the first-wave 50k, 100k, 150k, and 200k checkpoints?

## Immutable input panels

This follow-up evaluates all completed generated samples from the two
length-128 offset campaigns. It does not train or resample any model.

- First wave: four variants at 50k, 100k, 150k, and 200k.
- Expanded wave: 20 variants at the completed 200k panel.
- Every variant/milestone panel contains 32 samples at each of lengths 64, 96,
  and 128.
- Total: 108 ESMFold shards and 3,456 sequences.

The committed panel specification records the authoritative source roots and
SHA-256 hashes of the existing sample-analysis artifacts. A compute-side gate
also checks every sampling manifest, FASTA/PDB pair, sequence length, and exact
sample count before GPU inference can start.

## Endpoint and analysis

Every generated sequence is refolded in BF16 with the cached ESMFold container.
The primary endpoint is strict designability: generated-versus-ESMFold CA
Kabsch RMSD below 2 A **and** mean ESMFold pLDDT above 80. Reports also retain
the two threshold rates separately, continuous RMSD, mean pLDDT, and a
residue-indexed TM-like score.

Uncertainty is assessed with a deterministic paired bootstrap over matched
`(length, sample_index)` samples. The four-arm trajectory uses
`ratio3_offset0p2` as its promoted compact-clock reference. The expanded and
combined terminal panels use `ratio1_control`.

## Execution design

One integrity audit and one immediate end-to-end ESMFold canary jointly gate
all 108 independent GPU shards. Four first-wave milestone reports feed a
trajectory report; separate expanded-wave and combined 24-arm reports cover
the terminal 200k checkpoint. This fills available GPUs without touching the
running length-256 training campaign.

## Decision summary

**Corrected result (2026-08-24): designability was substantially understated.**
The original ESMFold wrapper averaged the full `[L, 37]` confidence tensor,
including zero-filled slots for atoms absent from each residue. Recomputing
mean pLDDT over atoms actually present in the predicted PDB changes every
pLDDT-dependent endpoint below; RMSD values are unchanged.

The corrected nominal leader is `ratio2_offset0p1_onset5`, with 28/96 strict
designable samples (29.17%; Wilson 95% interval 21.02%-38.92%). The no-delay
`ratio1_control` has 18/96 (18.75%). The observed difference is +10.42
percentage points, with a paired-bootstrap 95% interval of 0.00 to +20.83
points. The leader receives 60.5% `P(best)`, but the interval touches zero, so
one panel does not establish a statistically decisive training-arm winner.

The failure modes are informative:

- Backbone self-consistency remains moderate: 14 arms have median CA RMSD
  below 2 A, and `ratio2_offset0p4` and `ratio2_offset0p1` each pass RMSD <2 A
  for 63.5% of samples.
- Confidence is still more selective than geometry, but not catastrophically
  so. The leader passes RMSD for 60.4%, pLDDT for 32.3%, and both for 29.2%; its
  corrected mean pLDDT is 70.99.
- The diversity-leading clock-gated lDDT arm has 13/96 (13.54%) designability,
  below the 18/96 control. Its sequence-diversity advantage still does not
  improve this endpoint.
- The combined lDDT/self-conditioning gate remains a clear failure: zero RMSD
  passes, median CA RMSD 19.04 A, and corrected mean pLDDT 47.88.
- Training maturity is important. The four first-wave arms rise from 0%-1% at
  50k to 9.4%-22.9% at 200k. The earlier claim that longer training did not
  improve strict designability is invalidated.

There is a strong length gradient. The maximum is 15/32 (46.88%) at length 64
for `ratio2_offset0p1`, 8/32 (25.0%) at length 96 for
`ratio2_offset0p1_onset5`, and 6/32 (18.75%) at length 128 for the same onset-5
arm. This is meaningful partial designability, but still below the corrected
released-Pallatom calibration recorded in the ledger-wide rescore.

## Complete 200k ranking

Each row contains 96 samples. `P(best)` is a paired-bootstrap selection
probability within this panel, not an absolute measure of designability.

| Rank | Arm | Designable | RMSD<2 | pLDDT>80 | Median RMSD (A) | Mean pLDDT | P(best) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `ratio2_offset0p1_onset5` | 28/96 (29.2%) | 60.4% | 32.3% | 1.637 | 70.99 | 60.5% |
| 2 | `ratio1p5_offset0p05` | 25/96 (26.0%) | 52.1% | 28.1% | 1.918 | 70.72 | 27.1% |
| 3 | `ratio2_offset0p2` | 22/96 (22.9%) | 59.4% | 24.0% | 1.729 | 71.03 | 5.0% |
| 4 | `ratio2_offset0p05` | 20/96 (20.8%) | 52.1% | 24.0% | 1.792 | 68.42 | 1.2% |
| 5 | `ratio2_offset0p1_cbdelayed` | 20/96 (20.8%) | 52.1% | 30.2% | 1.917 | 69.41 | 2.3% |
| 6 | `ratio2_offset0p1` | 19/96 (19.8%) | 63.5% | 24.0% | 1.647 | 69.35 | 0.6% |
| 7 | `ratio2_offset0p4` | 18/96 (18.8%) | 63.5% | 20.8% | 1.638 | 69.63 | 0.8% |
| 8 | `ratio3_offset0p1_onset5` | 18/96 (18.8%) | 54.2% | 21.9% | 1.803 | 68.45 | 0.4% |
| 9 | `ratio3_offset0p1_cbdelayed` | 18/96 (18.8%) | 51.0% | 19.8% | 1.980 | 67.10 | 0.7% |
| 10 | `ratio3_offset0p05` | 18/96 (18.8%) | 49.0% | 19.8% | 2.050 | 67.68 | 0.5% |
| 11 | `ratio1_control` | 18/96 (18.8%) | 46.9% | 25.0% | 2.137 | 68.41 | 0.5% |
| 12 | `ratio3_offset0p1_lddt_narrow` | 17/96 (17.7%) | 49.0% | 24.0% | 2.048 | 69.67 | 0.4% |
| 13 | `ratio2_offset0p2_cbdelayed` | 15/96 (15.6%) | 58.3% | 17.7% | 1.764 | 69.67 | 0.0% |
| 14 | `ratio4_offset0p05` | 15/96 (15.6%) | 44.8% | 18.8% | 2.089 | 65.55 | 0.0% |
| 15 | `ratio1p5_offset0p4` | 14/96 (14.6%) | 55.2% | 17.7% | 1.799 | 67.79 | 0.0% |
| 16 | `ratio3_offset0p1` | 14/96 (14.6%) | 50.0% | 20.8% | 1.982 | 67.66 | 0.0% |
| 17 | `ratio3_offset0p1_lddt` | 13/96 (13.5%) | 44.8% | 17.7% | 2.212 | 68.29 | 0.0% |
| 18 | `ratio3_offset0p1_selfcond` | 12/96 (12.5%) | 44.8% | 15.6% | 2.110 | 64.92 | 0.0% |
| 19 | `ratio3_offset0p1_onset20` | 11/96 (11.5%) | 59.4% | 15.6% | 1.748 | 68.65 | 0.0% |
| 20 | `ratio3_offset0p2` | 9/96 (9.4%) | 53.1% | 15.6% | 1.864 | 68.01 | 0.0% |
| 21 | `ratio2_offset0p1_onset20` | 9/96 (9.4%) | 52.1% | 10.4% | 1.948 | 67.30 | 0.0% |
| 22 | `ratio3_offset0p2_cbdelayed` | 9/96 (9.4%) | 39.6% | 12.5% | 2.217 | 64.14 | 0.0% |
| 23 | `ratio3_offset0p1_lddt_wide` | 8/96 (8.3%) | 38.5% | 15.6% | 2.313 | 67.90 | 0.0% |
| 24 | `ratio3_offset0p1_both` | 0/96 (0.0%) | 0.0% | 1.0% | 19.038 | 47.88 | 0.0% |

## First-wave training trajectory

| Arm | 50k design. | 100k | 150k | 200k | 50k RMSD<2 | 200k RMSD<2 | 50k mean pLDDT | 200k mean pLDDT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ratio2_offset0p1` | 0.0% | 4.2% | 12.5% | 19.8% | 10.4% | 63.5% | 53.46 | 69.35 |
| `ratio2_offset0p2` | 1.0% | 5.2% | 13.5% | 22.9% | 13.5% | 59.4% | 55.18 | 71.03 |
| `ratio3_offset0p1` | 1.0% | 4.2% | 13.5% | 14.6% | 2.1% | 50.0% | 51.95 | 67.66 |
| `ratio3_offset0p2` | 0.0% | 4.2% | 9.4% | 9.4% | 8.3% | 53.1% | 50.64 | 68.01 |

## Interpretation limits

- These are fixed-panel comparisons from one training seed and one sampling
  seed, with 96 samples per arm. The paired bootstrap quantifies sampling-panel
  uncertainty, not training-seed variability.
- ESMFold self-consistency is a computational designability surrogate. It is
  not experimental evidence that a sequence expresses or folds, and its pLDDT
  calibration may differ across unusual generated sequence distributions.
- Twenty-four arms are compared, so nominal rank and `P(best)` should not be
  read as a confirmatory multiple-testing result. The control-relative interval
  for the leading arm crosses zero.
- The strict pLDDT >80 threshold is intentionally demanding. Correct Atom37
  masking is part of the endpoint definition; averaging absent atom slots is
  not a conservative alternative and must not be used.

## Launch record

### Canonical-pLDDT correction

The corrected table was generated by source commit
`df86846e19847c33e01109db5f63eddffaf28d54` in Scruffy workflow
`hk-esmfold-ledger-audit-df86846-v1`, job `job-9c2bca96039f3f2235b2`.
The read-only audit parsed the B factors in every predicted PDB and completed
with zero panel errors. Its immutable artifacts have SHA-256 values
`618c2404aa599c028636295c5ae1832fc106816088fa2a456c8522369a5a1ffb`
(`audit.json`) and
`77879cd18e45be7821fdacad5017939fdb691e08efd0fce073abca3ee5c4d320`
(`rows.jsonl`). The original artifacts remain unchanged for provenance.

The first launch used source commit `4abf61b6c9bf2220d0e9eca622bd7c2eaf65f175`
and workflow
`hk-delayed-sidechain-offset-length128-esmfold-4abf61b-v1`. Its inventory gate
validated all 3,456 inputs and its ESMFold canary succeeded, but the shared
Lustre filesystem reached 100% utilization during production inference.
Multiple otherwise-independent shards failed with `ENOSPC` while writing
predicted PDB files, so aggregate analyses were correctly skipped rather than
accepting incomplete panels.

Attempt 2 used source commit `bae385446c66be42c44a89a3093c056e13fa3ede`
and wrote ESMFold outputs to the capacious GBI shared filesystem.

The attempt-2 audit application validated the full input panel and exited zero,
but Scruffy subsequently marked its CPU-only Slurm placement invalid because
the step could not be reconciled. Production tasks therefore remained gated.
Attempt 3 keeps GBI output and requests one GPU for the short inventory gate so
the scheduler must authenticate a standard GPU placement before fan-out.

Attempt 3 completed successfully:

- Source commit: `5af33dd01f357d4df983f807ca33af03bbfb3d1f`
- Workflow: `hk-delayed-sidechain-offset-length128-esmfold-5af33dd-attempt3-v1`
- Allocation: `285372`
- Output root:
  `/mnt/gbi-shared/home/kiarash-jamali/hierarchical-kaveh-runs/delayed-sidechain-offset-length128-esmfold/5af33dd01f357d4df983f807ca33af03bbfb3d1f/attempt3/v1`
- Inventory gate: `job-b7f2ea4e49226915a567`, succeeded with all 3,456
  inputs and an authenticated placement.
- ESMFold canary: `job-5a1c9dd9f20b48e8f91a`, succeeded.
- Combined analysis: `job-571058385b851f545423`, succeeded.
- First-wave campaign analysis: `job-8a7e98aa5b616b41f7dd`, succeeded.
- Final workflow state: 117 succeeded; zero failed, skipped, rejected, queued,
  running, or blocked.

Result artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Combined 24-arm 200k JSON | `f863228a6bc1296cad7d0534a24bd291328f199c1f2e19d3db0e8e89022535ff` |
| Expanded-wave 200k JSON | `144487bac26868360841112dab5266033f84354cef819c3396d5e0bafdba0691` |
| First-wave campaign JSON | `8d9f7bacb36f25592f7741d8ae1b5c97a6e7e8880c59767826ae4ae3dc138c04` |
| First-wave 50k JSON | `1ac41882af52843946270d2252ddc3e3913fb383e4aa139b8b8f1a0d480458cb` |
| First-wave 100k JSON | `0be94f43f59b5d71862c20331b6c03d9a2dd6e24b317134b39cf62b85c2cca6a` |
| First-wave 150k JSON | `9bd9db44b17ad14c0dafb879baeb7245f86658ebd93c11dd9d5f0c8c407d55d9` |
| First-wave 200k JSON | `7f984f28faa417b773869602f3cf1625d7bb77ae34e5fb3009ae77fadca81173` |
