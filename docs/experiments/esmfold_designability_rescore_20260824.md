# Canonical ESMFold designability rescore (2026-08-24)

## Decision summary

The historical ESMFold evaluator understated confidence by averaging all 37
Atom37 slots, including zero-filled slots for atoms that do not exist in a
residue. The canonical endpoint instead averages pLDDT over `ATOM` records
actually present in each predicted PDB. CA RMSD, generated structures,
sequences, sampling metrics, and clash metrics were unaffected; every stored
mean-pLDDT value, pLDDT >80 pass, and strict-designability result was suspect.

The corrected evidence changes the working diagnosis:

1. **The local Pallatom evaluation is calibrated.** Released Pallatom reaches
   29/32 (90.6%) strict designability at its central `gamma=0.2`, step-scale
   2.25 setting and 30/32 (93.8%) at scale 3.25. The earlier 25% ceiling was an
   evaluator artifact, not a Pallatom sampler failure.
2. **Mature Kaveh is substantially better than the old ledger stated, but the
   raw co-design gap remains.** The 400k batch-256 legacy-recycling model rises
   monotonically to 35/96 (36.5%) at step scale 2.5. This is meaningful
   designability, but still far below released Pallatom on its reference panel.
3. **Many Kaveh backbones are sequence-saveable.** ProteinMPNN at scale 2.25
   gives 110/128 (85.9%) independently evaluated redesigns and 31/32 (96.9%)
   source backbones with at least one successful design among four. The latter
   is an oracle-over-four diagnostic, not an independent-sample rate.
4. **Training maturity matters.** In the four original offset arms, corrected
   strict designability rises from 0%-1% at 50k to 9.4%-22.9% at 200k. The old
   claim that longer training did not improve designability is invalid.

Taken together, the dominant gap is now **raw sequence/structure co-design and
sampler-scale calibration**, not evidence that nearly all generated backbones
are intrinsically unfoldable. The backbone is not fully exonerated: direct
Kaveh success remains 36.5% at best and falls strongly with length.

## Released Pallatom calibration

These are 32-sample length-100 panels using the released checkpoint. The paper
percentages are the corresponding values reported for the same hyperparameter
screen; finite panel and seed differences are expected.

| Gamma | Step scale | Old evaluator | Corrected strict | RMSD <2 A | Corrected mean pLDDT | Paper |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 1.75 | 0/32 (0.0%) | 17/32 (53.1%) | 25/32 | 78.03 | 57% |
| 0.2 | 2.25 | 4/32 (12.5%) | 29/32 (90.6%) | 32/32 | 84.80 | 87% |
| 0.1 | 2.25 | 7/32 (21.9%) | 23/32 (71.9%) | 28/32 | 83.63 | 89% |
| 0.2 | 2.75 | 8/32 (25.0%) | 29/32 (90.6%) | 30/32 | 85.18 | 94% |
| 0.2 | 3.25 | 7/32 (21.9%) | 30/32 (93.8%) | 30/32 | 87.49 | 93% |

An independent released-checkpoint length screen gives 26/32 at length 64,
28/32 at length 96, and 25/32 at length 128. The coherent-initial-noise
candidate and paper-literal two-pass self-conditioning candidate each give
28/32; neither beats the ordinary release screen. The separate timing screen
uses a different seed/protocol and must not be pooled with this table.

## Mature Kaveh step-scale response

This exact seven-scale panel uses the later batch-256 legacy-recycling
checkpoint (400k) with 96 samples per scale. Higher step scale remains strongly
favoured through the tested upper endpoint.

| Step scale | Old evaluator | Corrected strict | RMSD <2 A | Corrected mean pLDDT |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0/96 | 0/96 (0.0%) | 1/96 | 40.52 |
| 1.25 | 0/96 | 2/96 (2.1%) | 13/96 | 52.43 |
| 1.50 | 1/96 | 5/96 (5.2%) | 26/96 | 60.69 |
| 1.75 | 2/96 | 18/96 (18.8%) | 47/96 | 68.69 |
| 2.00 | 4/96 | 18/96 (18.8%) | 55/96 | 70.32 |
| 2.25 | 8/96 | 28/96 (29.2%) | 60/96 | 73.45 |
| 2.50 | 5/96 | 35/96 (36.5%) | 67/96 | 74.97 |

The earlier comprehensive checkpoint panel also promotes the batch-256
legacy-recycling arm (33/96 at scale 2.5) over the matched batch-256 control
(21/96). Both batch-32 legacy/control arms remain 0/96 at scale 2.5. The large
batch effect is therefore real on these matched panels and was hidden, not
created, by the confidence correction.

## ProteinMPNN backbone rescue

Each scale uses source sample indexes 0-31 from the mature Kaveh panel and four
ProteinMPNN sequences per fixed backbone. `Direct` is the original co-designed
Kaveh sequence on exactly those 32 backbones. `Per sequence` is the independent
redesign rate. `Best of four` asks whether any of four redesigns succeeds.

| Scale | Direct | ProteinMPNN per sequence | Best of four | Rescued | Lost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0/32 (0.0%) | 8/128 (6.2%) | 3/32 (9.4%) | 3 | 0 |
| 1.25 | 1/32 (3.1%) | 58/128 (45.3%) | 23/32 (71.9%) | 22 | 0 |
| 1.50 | 0/32 (0.0%) | 75/128 (58.6%) | 23/32 (71.9%) | 23 | 0 |
| 1.75 | 10/32 (31.2%) | 96/128 (75.0%) | 28/32 (87.5%) | 18 | 0 |
| 2.00 | 5/32 (15.6%) | 100/128 (78.1%) | 30/32 (93.8%) | 25 | 0 |
| 2.25 | 11/32 (34.4%) | 110/128 (85.9%) | 31/32 (96.9%) | 20 | 0 |
| 2.50 | 10/32 (31.2%) | 107/128 (83.6%) | 29/32 (90.6%) | 19 | 0 |

The peak MPNN setting is scale 2.25 rather than 2.5. This suggests that the
highest raw Kaveh step scale improves the jointly generated sequence but is
slightly less favourable for inverse-folding rescue than the adjacent scale.
No directly successful backbone is lost under best-of-four redesign.

## Length-128 delayed-clock panel

The complete 24-arm ranking and first-wave checkpoint trajectory are corrected
in [the dedicated ledger entry](delayed_sidechain_offset_length128_esmfold.md).
The new leader is `ratio2_offset0p1_onset5` at 28/96 (29.2%), versus 18/96
(18.8%) for `ratio1_control`. The paired-bootstrap difference is +10.4
percentage points with a 95% interval of 0.0 to +20.8 points, so this is a
numerical leader rather than a decisive training-arm win.

## Historical campaign inventory

The GBI audit covers all 409 discovered ESMFold panels and 21,907 predictions
with zero panel errors. The pooled counts below are an inventory, not one
binomial experiment: campaigns contain different panel sizes, deliberate
hyperparameter sweeps, and reused controls. Exact panel rows and duplicate
signatures are retained in the machine-readable ledger artifact.

| Campaign | Unique panels | Predictions | Old strict | Corrected strict | Best corrected panel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current sampler step scale | 91 | 8,736 | 82 | 510 | 35/96 (36.5%) |
| L128 offset clocks | 108 | 3,456 | 60 | 436 | 15/32 (46.9%) length slice |
| Delayed-sidechain sampler screen | 13 | 480 | 23 | 111 | 46/64 (71.9%), MPNN |
| EDM causal screen | 18 | 1,728 | 33 | 180 | 59/96 (61.5%), MPNN |
| EDM two-stage screen | 12 | 1,152 | 78 | 325 | 75/96 (78.1%), MPNN |
| Fixed quadrature checkpoints | 32 | 1,024 | 6 | 17 | 6/32 (18.8%) |
| Legacy quadrature late milestones | 8 | 256 | 4 | 14 | 6/32 (18.8%) |
| Legacy quadrature maturation | 56 | 1,792 | 1 | 16 | 7/32 (21.9%) |
| Legacy quadrature noise caps | 8 | 256 | 0 | 0 | 0/32 |
| Pallatom initial-noise screen | 1 | 32 | 10 | 28 | 28/32 (87.5%) |
| Pallatom reference lengths | 3 | 96 | 6 | 79 | 28/32 (87.5%) |
| Pallatom release hyperparameters | 5 | 160 | 26 | 128 | 30/32 (93.8%) |
| Pallatom two-pass screen | 1 | 32 | 2 | 28 | 28/32 (87.5%) |
| Pallatom timing screen | 3 | 96 | 4 | 28 | 12/32 (37.5%) |
| Obsolete ProteinMPNN wrapper | 7 | 896 | unavailable | unavailable | no adjacent RMSD rows |
| Sampler recurrence | 5 | 320 | 4 | 30 | 9/64 (14.1%) |
| Early step-scale screen | 7 | 224 | 3 | 15 | 6/32 (18.8%) |
| ProteinMPNN step-scale rescue | 7 | 896 | 181 | 554 | 110/128 (85.9%) |

Within-campaign duplicate panels are counted once in this inventory. In
particular, mirrored legacy-quadrature controls do not create extra evidence.

## What remains valid from earlier analyses

- All CA RMSD rankings, clash measurements, sequence-composition metrics, and
  visual structure assessments remain numerically valid.
- Stronger step scale helps mature Kaveh sampling; that conclusion becomes
  stronger after correction.
- Extra side-chain noise caps can still harm legacy-quadrature sampling. The
  zero-success cap screen remains zero after correction.
- The combined lDDT/self-conditioning offset arm remains a geometric failure.
- Raw Kaveh remains below released Pallatom, although the gulf is much smaller.

The invalidated claims are those based on the old pLDDT averages: that no arm
had material designability, that training maturity barely helped, that
ProteinMPNN could not rescue most backbones, and that the released Pallatom
checkpoint failed to reproduce its paper.

## Provenance and endpoint contract

The evaluator correction is source commit
`9c993afa1ac77448d0df3d0d0abbcb0f8b7e2e65`. The GBI read-only audit used
commit `df86846e19847c33e01109db5f63eddffaf28d54`, workflow
`hk-esmfold-ledger-audit-df86846-v1`, and job
`job-9c2bca96039f3f2235b2`. It completed with zero errors. Audit artifact hashes:

- `audit.json`: `618c2404aa599c028636295c5ae1832fc106816088fa2a456c8522369a5a1ffb`
- `rows.jsonl`: `77879cd18e45be7821fdacad5017939fdb691e08efd0fce073abca3ee5c4d320`

Historical output directories were not modified. Strict designability remains
defined as CA Kabsch RMSD <2 A **and** canonical mean predicted-atom pLDDT >80.
It is an ESMFold self-consistency surrogate, not experimental fold validation.
