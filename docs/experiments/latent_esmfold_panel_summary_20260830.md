# ESMFold panel summaries: selected-4, scratch, and filtered recovery

**Snapshot:** 2026-08-30  
**Endpoint:** durable `panel_summary.json` plus PDB outputs  
**Designability definition:** CA RMSD < 2 A **and** mean pLDDT > 80

## Scope and provenance

The machine-readable artifact
[`artifacts/latent_esmfold_panel_summary_20260830.csv`](artifacts/latent_esmfold_panel_summary_20260830.csv)
contains **101 aggregate panels and 363 length cells**:

- 86 filtered all-16 panels, from 50k through 400k, code commit
  `b81485f257299928492ebc38a31e0eccb477fb02`;
- 12 selected-4 L256 panels, 250k through 350k, code commit
  `b6ccee755b9420385ea09a132378e8a7af6d4193`;
- 3 scratch-from-zero L256 panels for `r1_z4_b0p03_c0p25`, 50k through 150k,
  the same `b6ccee755b9420385ea09a132378e8a7af6d4193` commit.

The matched visualization is
[`artifacts/latent_esmfold_panel_summary_20260830.png`](artifacts/latent_esmfold_panel_summary_20260830.png).
The authoritative remote roots are:

- filtered: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-filtered-all16-400k/b81485f257299928492ebc38a31e0eccb477fb02/recovery-esmfold-v2/esmfold`;
- selected-4: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-selected4-length256-500k-b256-recovery/b6ccee755b9420385ea09a132378e8a7af6d4193/esmfold`;
- scratch: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-r1_z4_b0p03_c0p25-length256-500k-b256-scratch-v2/b6ccee755b9420385ea09a132378e8a7af6d4193/esmfold`.

## Selected-4 and scratch results

| Campaign | Step | Arm | N | Designability | Mean pLDDT | RMSD <2 A | Mean pTM |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| selected-4 | 250k | `r0_z3_b0p3_c0p125` | 224 | 23.7% | 70.72 | 53.1% | 0.789 |
| selected-4 | 250k | `r1_z3_b0p01_c0p125` | 224 | 24.6% | 70.73 | 50.9% | 0.779 |
| selected-4 | 250k | `r1_z4_b0p03_c0p25` | 224 | 30.8% | 75.77 | 62.1% | 0.845 |
| selected-4 | 250k | `r2_z2_b0p1_c0p125` | 224 | 14.3% | 61.55 | 31.2% | 0.507 |
| selected-4 | 300k | `r0_z3_b0p3_c0p125` | 224 | 33.0% | 70.17 | 50.0% | 0.738 |
| selected-4 | 300k | `r1_z3_b0p01_c0p125` | 224 | 28.6% | 71.06 | 49.1% | 0.764 |
| selected-4 | 300k | `r1_z4_b0p03_c0p25` | 224 | 37.5% | 78.73 | 67.4% | 0.866 |
| selected-4 | 300k | `r2_z2_b0p1_c0p125` | 224 | 19.2% | 60.59 | 30.4% | 0.634 |
| selected-4 | 350k | `r0_z3_b0p3_c0p125` | 224 | 36.6% | 73.59 | 52.2% | 0.790 |
| selected-4 | 350k | `r1_z3_b0p01_c0p125` | 224 | 43.3% | 78.36 | 61.6% | 0.839 |
| selected-4 | 350k | `r1_z4_b0p03_c0p25` | 224 | **46.4%** | 78.96 | 76.8% | 0.881 |
| selected-4 | 350k | `r2_z2_b0p1_c0p125` | 224 | 17.4% | 55.49 | 32.6% | 0.575 |
| scratch | 50k | `r1_z4_b0p03_c0p25` | 224 | 3.6% | 65.42 | 35.3% | 0.782 |
| scratch | 100k | `r1_z4_b0p03_c0p25` | 224 | 15.2% | 72.81 | 61.2% | 0.841 |
| scratch | 150k | `r1_z4_b0p03_c0p25` | 224 | 20.1% | 73.75 | 50.9% | 0.827 |

The selected-4 `r1_z4_b0p03_c0p25` arm is the clear current leader, improving
from 30.8% at 250k to 46.4% at 350k. The `r1_z3` arm also improves strongly
(24.6% to 43.3%). `r0_z3` improves more gradually, while `r2_z2` remains
weak and non-monotonic. The scratch control improves with training but is
still below the resumed arm at the same nominal 150k/250k range.

Length remains the principal limitation. For selected-4 `r1_z4`, the 250k
length cells are 62.5% at L64, 40.6% at L96, 28.1% at L128, 25.0% at L160,
28.1% at L192, 31.2% at L224, and 0.0% at L256. By 350k, L256 reaches 37.5%,
but the aggregate is still only 46.4%; short-length success must not be
mistaken for 100% designability at L256.

## Filtered all-16 result

At each checkpoint the best and worst filtered arms were:

| Step | Best arm / designability | Worst arm / designability |
| ---: | --- | --- |
| 50k | `r1_z2_b0p3_c0p25` / 42.7% | `r0_large_z8_b0p01_c0p125` / 12.5% |
| 100k | `r0_z5_b0p03_c0` / 42.7% | `r0_large_z8_b0p01_c0p125` / 27.1% |
| 150k | `r0_z2_b0p1_c0` / 40.6% | `r1_z4_b0p3_c0p5` / 24.0% |
| 200k | `r0_z2_b0p1_c0` / 38.5% | `r0_z3_b0p3_c0p125` / 13.5% |
| 250k | `r1_z3_b0p01_c0p125` / 38.5% | `r2_z4_b0p03_c0p25` / 9.4% |
| 300k | `r0_z2_b0p1_c0` / 39.6% | `r1_z3_b0p01_c0p125` / 7.3% |
| 350k | `r0_large_z4_b0p01_c0p25` / 33.3% | `r0_z3_b0p03_c0p5` / 12.5% |
| 400k | `r0_z3_b0p3_c0p125` / 31.2% | `r0_large_z8_b0p01_c0p125` / 9.4% |

Filtering does not produce a uniform improvement with checkpoint. The best
filtered arm never exceeds 42.7% in these panels, and late-step performance
is often lower than the early filtered results. Rankings are arm- and
checkpoint-specific, so the next comparison should use matched lengths and
the same sampler schedule rather than one global filtered ranking.

## Failure correction and interpretation

The selected-4 and scratch all-atom aggregator tasks that failed were not
ESMFold failures. They invoked the legacy
`analyze_latent_all_atom_rmsd.py`, which expects
`shard-*/.../per_sample.csv`; the current ESMFold workflow emits
`panel_summary.json` plus PDBs. The current `panel_summary.json` outputs are
therefore the authoritative foldability/designability endpoint for these
campaigns. The old aggregator failures are recorded as an interface mismatch,
not as evidence that the structures failed.

