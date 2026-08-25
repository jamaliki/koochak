# Recycling, sampling, and geometry lessons (2026-08-25)

Status: mixed; sampler bug fixed, throughput characterized, relative-position
factorial complete, steric trajectories complete but not canonically analyzed.

This entry collects results from the chat that do not belong solely to the
inverse-folding or signal-propagation campaigns. It is intentionally a map to
the detailed evidence rather than a replacement for the canonical rescore.

## Hierarchy-native recycling is feasible but not free

The implemented hierarchy predicts and recycles at each native resolution:

- coarse blocks predict the masked mean CA position of each four-residue patch
  and update a compact distance/RBF pair representation;
- residue blocks predict per-residue CA coordinates and sequence;
- atom blocks predict cumulative Atom14 coordinates and sequence;
- intermediate coordinate and sequence heads receive discounted supervision.

The throughput probe omitted Kabsch for intermediate targets but retained the
then-configured terminal alignment. Subsequent production recycling campaigns
also disabled terminal coordinate alignment, following the later experiment
decision. The detailed timing history, including the invalid asynchronous
first timer and the compile fallback repair, is in
[the hierarchy recycling throughput probe](hierarchy_recycling_throughput.md).

The compile-clean synchronized batch-32/H100 result was:

| Arm | p50 step | Versus baseline | p50 samples/s | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 136.64 ms | - | 234.19 | 9,617.5 MiB |
| Coarse recycling | 140.84 ms | +3.08% | 227.20 | 9,638.0 MiB |
| All recycling | 153.51 ms | +12.34% | 208.46 | 9,853.3 MiB |

Adding all 14 intermediate losses moved the all-recycling p50 from 154.13 to
160.05 ms (+3.85%) in the isolated pair. Matched steps put the typical increment
near 1-2 ms; the unpaired p95 was tail-heavy. Total recycling plus supervision
used only about 238 MiB more peak memory than baseline. The main steady-state
cost is architectural recycling, not intermediate loss evaluation.

No designability conclusion follows from the throughput screen. The most
credible next optimization remains a fused one-point patch RBF projection,
followed by a measured `addmm` fusion of narrow residue/atom feedback
projections. Compile-cache warmup is a larger operational issue for short jobs
than memory capacity.

## Quadrature self-conditioning alignment bug

The first old-Kaveh-style delayed sampler used a backbone-only mask to fit a
rigid alignment. The alignment helper also applied that mask to its returned
coordinates. Atom14 slots 5:14 were therefore zeroed while
`self_conditioning_mask = True`.

Training never sees that state: its coordinate self-conditioning contains all
valid Atom14 coordinates. Every nonzero side-chain-noise cap entered the faulty
branch, while cap 0 bypassed it, making additional side-chain noise look
catastrophic.

The repair separates the fit mask from the output mask: fit on stable backbone
atoms, transform and retain all atoms. Commit `a0f0bd7` contains the original
fix and regression test; the integrated sampler descendants retain the same
contract.

The complete 67-task batch/maturity/cap panel succeeded after the repair. The
most diagnostic batch-256, 160k slice was:

| Cap | Broken CA RMSD | Fixed CA RMSD | Broken clashes/res | Fixed clashes/res |
| ---: | ---: | ---: | ---: | ---: |
| 0 A | 4.40 | 4.40 | 0.225 | 0.225 |
| 3 A | 20.38 | 5.15 | 3.420 | 0.230 |
| 5 A | 19.16 | 4.80 | 2.954 | 0.241 |
| 8 A | 18.72 | 4.34 | 2.164 | 0.277 |

RMSD and clash metrics are unaffected by the later Atom37 pLDDT masking
correction. Historical pLDDT and strict-designability numbers from this panel
must not be reused without canonical rescoring.

After the fix, cap 8 was numerically best for foldability in this small panel
but increased clashes relative to cap 0 and lies outside the training support.
Training uses `e = 0` on half of examples and otherwise
`e ~ Uniform(0, 5 A)` in quadrature. The closest deterministic sampling cap is
therefore 5 A. Cap 8 is an extrapolative sampler intervention, not a faithful
training match.

## Canonical normal sampler and scale response

Unless an entry explicitly says otherwise, the matched normal panels in this
work use:

- 200 Euler steps;
- constant step scale 2.25;
- churn `gamma = 0.2`;
- EMA weights;
- legacy side-chain cap 5 A;
- recurrent recycling enabled;
- matched sample seeds.

The current-sampler step-scale sweep showed that mature batch-256 Kaveh kept
improving through the tested endpoint: corrected strict L128 designability was
0/96 at scale 1.0, 18/96 at 1.75, 28/96 at 2.25, and 35/96 at 2.5. ProteinMPNN
backbone rescue peaked one point earlier at scale 2.25. Full tables and the
recovered old-Kaveh sampler comparison are in
[the canonical corrected rescore](esmfold_designability_rescore_20260824.md).

The recovered old structural sampler changed a no-tricks model from 2/48 to
26/48 strict designs, but it jointly changed steps, rho, sigma endpoints, step
scale schedule, and recurrence windows. It localizes the missing effect to the
sampler bundle; it does not identify one causal setting.

## Four-arm batch and quadrature production campaign

The main from-scratch recycling campaign crossed batch size with the accepted
quadrature side-chain corruption:

| Arm | Batch | Quadrature training |
| --- | ---: | --- |
| `batch32_control` | 32 | No |
| `batch32_legacy_quadrature` | 32 | 50% `Uniform(0, 5 A)` extra sigma |
| `batch256_control` | 256 | No |
| `batch256_legacy_quadrature` | 256 | 50% `Uniform(0, 5 A)` extra sigma |

All four LR 3e-4 trajectories reached 400k. Workflow and commit provenance are
`hk-recycling-quadrature-batch-4x400k-6cefa20-v1` and
`6cefa205db6be4361351191e19b393e1bc75e523`.

Corrected endpoint analysis shows that batch size was the stronger effect. At
step scale 2.5, batch-256 legacy quadrature reached 33/96 strict designs versus
21/96 for the matched batch-256 control, while both batch-32 arms remained
0/96. This supports the batch-256 and quadrature combination for later work,
but does not isolate whether the large-batch advantage comes from optimization,
sample exposure, or length-mixture statistics.

A matched LR 1e-3 repeat also reached mature checkpoints, but its activation
audit found severe finite scale collapse, including content RMS near 3e15 in
the batch-256 control. Those checkpoints are rejected regardless of occasional
small-panel geometry; see
[the signal-propagation ledger](signal_propagation_and_recycling_20260825.md#fully-trained-baseline-audit).

## Backbone clashes are predominantly oxygen-backbone contacts

The expanded clash analysis separates nonlocal backbone-backbone,
oxygen-backbone, side-chain, and all-atom contacts. In mature panels, virtually
all counted backbone clashes involve a backbone oxygen:

| Model/panel | BB clashes/res | O-BB clashes/res |
| --- | ---: | ---: |
| Signal pair, 200k | 0.059 | 0.058 |
| Signal pair, 250k | 0.057 | 0.056 |
| Signed-log RoPE, 400k | 0.072 | 0.072 |
| AF2 clip30 + RoPE, 400k | 0.058 | 0.058 |

This points to local peptide-plane and nonlocal oxygen placement as a more
specific problem than generic CA-chain continuity: the same panels have
approximately zero bad CA steps. A side-chain-only packing phase cannot repair
this backbone error.

A mild low-sigma steric objective was implemented with weight 0.1, sigma gate
0.5 A, overlap tolerance 0.4 A, and eight nearest nonbonded candidates. It was
applied to terminal and recycled atom predictions. The repaired workflow
`hk-recycling-steric-b256-2x400k-2f900a8-v3` completed both batch-256 400k
trajectories and all 27 tasks. No corrected Atom37 ESMFold aggregate was found
in that workflow, so the steric intervention is **trained but not canonically
evaluated** and must not be promoted from training completion alone.

The next steric analysis should report O-BB separately, compare matched control
and candidate checkpoints at equal maturity, and verify that any reduction does
not merely expand the backbone or lower contact density.

## AF2-style clipped relative residue index did not help

The coarse trunk already had patch-start RoPE and a signed-log sequence-offset
feature. A three-arm batch-256 factorial compared:

1. `signed_log_rope` (baseline);
2. AF2-style relative residue index clipped at +/-30 with no RoPE;
3. the clipped feature plus RoPE.

All three reached 400k. The corrected, matched 32-sample L128 endpoint was:

| Arm | Strict | Mean RMSD | Mean pLDDT | BB/O-BB clashes | Eff. alphabet | Ala |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Signed-log + RoPE | **21.9%** | 4.11 A | 69.7 | 0.072 / 0.072 | 8.43 | 27.7% |
| AF2 clip30, no RoPE | 6.2% | 5.45 A | 65.1 | 0.096 / 0.096 | 8.20 | 29.7% |
| AF2 clip30 + RoPE | 12.5% | 3.79 A | 66.9 | 0.058 / 0.058 | 8.24 | 31.2% |

Relative to baseline, the strict-designability effects were -15.6 points
(bootstrap 95% interval -34.4 to 0.0) without RoPE and -9.4 points (-28.1 to
+9.4) with RoPE. Neither arm improved designability; the no-RoPE arm was clearly
worse. The +RoPE arm reduced oxygen-backbone clashes numerically, but its
interval included no change and it increased alanine and homopolymer runs. Do
not promote the AF2 clip30 replacement.

Provenance: workflow `hk-coarse-relpos-b256-3x400k-4a7bbb4-v1`, commit
`4a7bbb4f7ab4462bbc5897b721bbd615332ed99c`, corrected summary source
`pdb_atom37_masked_b_factors`.

## Length bucketing remains an engineering assumption

The production L128 campaigns use buckets 64/96/128 with patch capacities
16/24/32. Fixed-shape L128 benchmarks established feasible batch sizes and
bucketed shapes reduce padding for mixed-length training, but this chat did not
run a matched training-quality comparison between three buckets and one L128
bucket.

Therefore there is no evidence yet that length bucketing improves model quality.
For a pure L128 campaign, one exact L128 bucket is a reasonable simplification
and compile-stability experiment; for future L256 training, two or three buckets
remain a throughput/padding choice that should be benchmarked, not treated as a
learned-model intervention.

## Evaluation rules that apply to every result

1. ESMFold mean pLDDT must average B-factors from actual Atom37 PDB `ATOM`
   records. Accept a summary only when
   `mean_plddt_source = pdb_atom37_masked_b_factors`.
2. CA RMSD, generated coordinates, sequences, and clash metrics were unaffected
   by the zero-filled-Atom37 confidence bug.
3. Four-sample activation or structure screens are diagnostics, not confidence
   intervals or designability estimates.
4. Best-of-four ProteinMPNN rescue measures whether a backbone can be saved by
   an oracle over four sequences; report the per-sequence rate separately.
5. Training completion, successful sampling, and corrected ESMFold analysis are
   distinct milestones. Never infer one from another.
