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

**No variant has high strict designability on this panel.** The nominal leader,
`ratio2_offset0p05`, has 6/96 designable samples (6.25%; Wilson 95% interval
2.90%-12.97%). The no-delay `ratio1_control` has 2/96 (2.08%). The paired
bootstrap difference is +4.13 percentage points, but its 95% interval is
-1.04 to +10.42 points, so this panel does not establish a statistically clear
improvement over the control.

The failure modes are informative:

- Backbone self-consistency is already moderate: 14 arms have median CA RMSD
  below 2 A, and `ratio2_offset0p4` and `ratio2_offset0p1` each pass RMSD <2 A
  for 63.5% of samples.
- High ESMFold confidence is the bottleneck. The best pLDDT >80 rate is only
  6.25%, and the best mean pLDDT over a variant is 65.40.
- The diversity-leading clock-gated lDDT arm has 2.08% designability, equal to
  the no-delay control. Its additional sequence diversity does not translate
  into higher ESMFold confidence on this fixed panel.
- The combined lDDT/self-conditioning gate remains a clear failure: zero RMSD
  passes, median CA RMSD 19.04 A, and mean pLDDT 46.31.
- Training from 50k to 200k strongly improves backbone self-consistency and
  mean pLDDT for every first-wave arm, but strict designability remains at only
  1.04%-2.08% at 200k. Longer training alone is therefore unlikely to close
  the confidence gap.

No length slice is highly designable either. The maximum is 3/32 (9.38%) at
length 64 (`ratio2_offset0p05` and `ratio3_offset0p1_onset5`), 2/32 (6.25%) at
length 96 (`ratio1_control` and `ratio3_offset0p1_onset20`), and 2/32 (6.25%)
at length 128 (`ratio2_offset0p05` and `ratio3_offset0p1_selfcond`).

## Complete 200k ranking

Each row contains 96 samples. `P(best)` is a paired-bootstrap selection
probability within this panel, not an absolute measure of designability.

| Rank | Arm | Designable | RMSD<2 | pLDDT>80 | Median RMSD (A) | Mean pLDDT | P(best) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `ratio2_offset0p05` | 6/96 (6.2%) | 52.1% | 6.2% | 1.792 | 63.27 | 45.2% |
| 2 | `ratio3_offset0p1_onset5` | 5/96 (5.2%) | 54.2% | 5.2% | 1.803 | 63.34 | 24.8% |
| 3 | `ratio4_offset0p05` | 4/96 (4.2%) | 44.8% | 4.2% | 2.089 | 60.96 | 11.1% |
| 4 | `ratio2_offset0p1_onset5` | 3/96 (3.1%) | 60.4% | 4.2% | 1.637 | 65.40 | 4.4% |
| 5 | `ratio3_offset0p1_onset20` | 3/96 (3.1%) | 59.4% | 4.2% | 1.748 | 63.44 | 4.4% |
| 6 | `ratio2_offset0p1_cbdelayed` | 3/96 (3.1%) | 52.1% | 4.2% | 1.917 | 64.16 | 0.5% |
| 7 | `ratio1p5_offset0p05` | 3/96 (3.1%) | 52.1% | 3.1% | 1.918 | 65.11 | 2.0% |
| 8 | `ratio3_offset0p05` | 3/96 (3.1%) | 49.0% | 3.1% | 2.050 | 62.60 | 2.3% |
| 9 | `ratio3_offset0p1_selfcond` | 3/96 (3.1%) | 44.8% | 3.1% | 2.110 | 60.45 | 2.8% |
| 10 | `ratio2_offset0p2` | 2/96 (2.1%) | 59.4% | 2.1% | 1.729 | 65.40 | 0.3% |
| 11 | `ratio3_offset0p2` | 2/96 (2.1%) | 53.1% | 2.1% | 1.864 | 62.81 | 0.1% |
| 12 | `ratio2_offset0p1_onset20` | 2/96 (2.1%) | 52.1% | 2.1% | 1.948 | 62.37 | 0.3% |
| 13 | `ratio3_offset0p1` | 2/96 (2.1%) | 50.0% | 2.1% | 1.982 | 62.68 | 0.4% |
| 14 | `ratio1_control` | 2/96 (2.1%) | 46.9% | 2.1% | 2.137 | 63.30 | 0.7% |
| 15 | `ratio3_offset0p1_lddt` | 2/96 (2.1%) | 44.8% | 2.1% | 2.212 | 63.12 | 0.8% |
| 16 | `ratio2_offset0p4` | 1/96 (1.0%) | 63.5% | 1.0% | 1.638 | 64.21 | 0.0% |
| 17 | `ratio2_offset0p1` | 1/96 (1.0%) | 63.5% | 2.1% | 1.647 | 63.96 | 0.0% |
| 18 | `ratio2_offset0p2_cbdelayed` | 1/96 (1.0%) | 58.3% | 1.0% | 1.764 | 64.21 | 0.0% |
| 19 | `ratio3_offset0p1_cbdelayed` | 1/96 (1.0%) | 51.0% | 1.0% | 1.980 | 62.13 | 0.0% |
| 20 | `ratio3_offset0p1_lddt_narrow` | 1/96 (1.0%) | 49.0% | 1.0% | 2.048 | 64.28 | 0.0% |
| 21 | `ratio3_offset0p2_cbdelayed` | 1/96 (1.0%) | 39.6% | 1.0% | 2.217 | 59.75 | 0.0% |
| 22 | `ratio1p5_offset0p4` | 0/96 (0.0%) | 55.2% | 0.0% | 1.799 | 62.60 | 0.0% |
| 23 | `ratio3_offset0p1_lddt_wide` | 0/96 (0.0%) | 38.5% | 0.0% | 2.313 | 62.77 | 0.0% |
| 24 | `ratio3_offset0p1_both` | 0/96 (0.0%) | 0.0% | 0.0% | 19.038 | 46.31 | 0.0% |

## First-wave training trajectory

| Arm | 50k design. | 100k | 150k | 200k | 50k RMSD<2 | 200k RMSD<2 | 50k mean pLDDT | 200k mean pLDDT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ratio2_offset0p1` | 0.0% | 2.1% | 1.0% | 1.0% | 10.4% | 63.5% | 51.32 | 63.96 |
| `ratio2_offset0p2` | 0.0% | 1.0% | 2.1% | 2.1% | 13.5% | 59.4% | 52.75 | 65.40 |
| `ratio3_offset0p1` | 0.0% | 0.0% | 2.1% | 2.1% | 2.1% | 50.0% | 50.29 | 62.68 |
| `ratio3_offset0p2` | 0.0% | 0.0% | 1.0% | 2.1% | 8.3% | 53.1% | 49.15 | 62.81 |

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
- The strict pLDDT >80 threshold is intentionally demanding. Reporting the
  RMSD and pLDDT components separately shows that confidence, rather than only
  geometric self-consistency, is the dominant failure mode.

## Launch record

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
