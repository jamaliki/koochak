# Fixed-sampler feature and architecture screen

## Executive summary

This note records the fixed-sampler reruns of the shorter feature and
architecture experiments that followed the stereochemistry factorial. The
reruns use the same trained checkpoints as the original screens; only the
sampling implementation changed. The fixed sampler uses a coherent perturbed-
time EDM grid and the recurrent self-conditioning frame expected by the
denoiser. It removes a sampling-time artifact that had made some pair-route
and sequence outputs collapse.

The current ranking, after the matched 100k continuations, is:

1. **Position/element + `sig05_uniform_atomsc`, 100k**: best balanced
   promotion candidate under the mature-checkpoint screen.
2. **Position/element + `sig10_polar2_atomsc`, 100k**: best raw sequence
   diversity, but with a higher clash rate.
3. **Alanine-reference features, 100k**: competitive and geometrically clean,
   but slightly behind the best position/element cells on sequence diversity.
4. **Window16 low-noise sequence-gate family, 25k**: sequence diversity is
   competitive, but clashes remain high.
5. **Window16 index and information-flow families**: no evidence of a better
   overall trade-off under the shared sampling contract.

The best-balanced recommendation is **position/element features with the
`sig05_uniform_atomsc` cell at 100k**. This is a 0.5 A sequence-loss gate with
uniform amino-acid class weighting and atom-coordinate self-conditioning. The
`sig10_polar2_atomsc` cell is the sequence-diversity leader, but its clash rate
is more than twice that of the recommended cell. The old standalone
low-noise-gate family is not the winner; its geometry remains unsafe.

## Scope and provenance

The campaigns covered here are the successful fixed-sampler reruns requested
for the earlier feature/architecture checkpoints:

| Campaign | Source checkpoint | Variants | Fixed-sampler result |
|---|---:|---:|---|
| Alanine-reference features | 25k | 4 | 4/4 sampling; analysis succeeded |
| Position/element features | 50k | 8 | 8/8 sampling; analysis succeeded |
| Window16 index features | 25k | 8 | 8/8 sampling; corrected analysis succeeded |
| Information-flow variants | 50k | 8 | 8/8 sampling; analysis succeeded |
| Low-noise sequence gate | 25k | 8 | 8/8 sampling; corrected analysis succeeded |
| Selected 100k continuations | 100k | 8 | 8/8 sampling; analysis succeeded |

The fixed sampler was prepared from the sampler-fix line based on commit
`e79921b`, with the coherent perturbed-time grid. The wrapper commits used for
the five output sets were:

- Alanine-reference: `c87f70f`
- Position/element: `851ce35`
- Window16 index: `29795f0`
- Information-flow: `ba33387`
- Low-noise sequence gate: `e7fbaa1`

The output manifests and aggregate analysis files are stored under the
campaign roots, for example:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  alanine-reference-25k-factorial/c87f70f/samples/step025000/analysis_fixed.json
  position-element-50k-factorial/851ce35/samples/step050000/analysis_fixed.json
  window16-index-25k-factorial/29795f0/samples/step025000/analysis_fixed.json
  flow-50k-factorial/ba33387/samples/step050000/analysis_fixed.json
  low-noise-seq-25k-factorial/e7fbaa1/samples/step025000/analysis_fixed.json
```

The selected 100k continuation panel was sampled from commit `890cbc2` under:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  top8-reference-sampling/890cbc2/step100000/
```

Each sampling panel used EMA weights, BF16, compile/fused execution, lengths
64/96/128, 32 samples per length, batch size 32, seed `20260813`, 200
perturbed-time Euler steps, the matching recurrent self-conditioning route,
`gamma=0.2`, churn interval `0.01 <= t <= 1.0` evaluated on the perturbed
normalized time, noise scale `1.003`, step scale `2.25`, and sequence
temperature `0.1`. Churn therefore remains active through the low-noise gate
until the perturbed time falls below `0.01`; the final Euler endpoint is not
passed through the denoiser. The trained model, optimizer, data, and
scientific settings were not changed for these reruns.

## Why the sampler fix matters

The original comparison mixed denoiser behavior with an implementation error
in the sampling time/self-conditioning path. The corrected sampler keeps the
time values used for corruption, denoising, and recurrent self-conditioning
consistent. This is especially important for the pair-distance route, where
the old implementation could turn a viable checkpoint into a collapsed or
geometrically implausible sample after many Euler steps.

The fixed-sampler comparison therefore answers a narrower and more useful
question: *given the same trained checkpoint, which feature or information-flow
choices produce the best outputs under a valid sampling procedure?* It should
not be read as a new training ablation.

## Aggregate results

The values below are means over the variants and sampled lengths in each
campaign. They are screening summaries, not per-variant promotion scores.

### Sequence and stability metrics

| Campaign | Step | Effective alphabet | Entropy | Max residue fraction | Max homopolymer run |
|---|---:|---:|---:|---:|---:|
| Alanine-reference | 25k | **5.794** | **0.569** | 0.482 | **7.245** |
| Position/element | 50k | 5.185 | 0.534 | **0.529** | 7.685 |
| Window16 index | 25k | 4.652 | 0.498 | 0.564 | 8.574 |
| Information-flow | 50k | 4.242 | 0.464 | 0.605 | 9.915 |
| Low-noise sequence gate | 25k | 4.973 | 0.520 | 0.545 | 8.456 |

Higher effective alphabet and entropy, lower maximum residue fraction, and
shorter maximum runs indicate less sequence collapse. On these measures the
alanine-reference family is the strongest, followed by position/element and
low-noise sequence gating. Information-flow variants are the weakest at the
matched 50k duration.

The alanine-reference lead should not be overinterpreted: it is measured at
25k, whereas position/element and information-flow are measured at 50k. A
25k checkpoint can look more diverse simply because it has not yet developed
the same degree of specialization or collapse.

### Backbone and geometry proxies

| Campaign | CA-step bad fraction | Peptide bad fraction | CA clashes |
|---|---:|---:|---:|
| Alanine-reference, 25k | 0.0087 | 0.0109 | 0.854 |
| Position/element, 50k | **0.0005** | **0.0013** | 0.616 |
| Window16 index, 25k | 0.0000 | 0.0013 | 1.039 |
| Information-flow, 50k | 0.0003 | 0.0013 | 0.876 |
| Low-noise sequence gate, 25k | 0.0139 | 0.0128 | **5.529** |

The position/element family has the best balanced geometry among the candidates
with useful sequence diversity. The Window16 index screen has low backbone
proxy violation rates but more clashes and lower sequence quality. The
low-noise gate is the clear geometry warning: its sequence statistics are
competitive, but its clash rate is an order of magnitude above the other
families.

## 100k continuation results

The table below reports the eight selected 100k checkpoints. Values are means
over lengths 64/96/128 and 32 samples per length. These are the first mature
results at a matched 100k horizon for the two continued families; the other
families remain represented by their earlier 25k or 50k screens above.

| Family / variant | Effective alphabet | Entropy | Max residue fraction | Max run | CA-step bad | Peptide bad | CA clashes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alanine / `baseline` | 5.729369 | 0.571873 | 0.488824 | 6.927083 | 0.000495 | 0.000000 | 0.166667 |
| Alanine / `element` | 5.350766 | 0.540211 | 0.535265 | 8.468750 | 0.000000 | 0.000000 | 0.177083 |
| Alanine / `position` | 5.377835 | 0.546705 | 0.516222 | 7.656250 | 0.000110 | 0.000000 | 0.114583 |
| Alanine / `position_element` | 4.490817 | 0.483629 | 0.594455 | 9.395833 | 0.000000 | 0.000000 | 0.208333 |
| Position/element / `sig05_uniform_atomsc` | 5.555960 | 0.551250 | 0.470486 | 7.062500 | 0.000411 | 0.000520 | 0.114583 |
| Position/element / `sig05_uniform_pairsc` | 5.067879 | 0.523159 | 0.547173 | 8.729167 | 0.000000 | 0.000000 | 0.135417 |
| Position/element / `sig10_polar2_atomsc` | **5.887323** | **0.576669** | **0.454102** | **6.322917** | **0.000082** | 0.000629 | 0.270833 |
| Position/element / `sig10_polar2_pairsc` | 4.740093 | 0.498230 | 0.581950 | 9.687500 | 0.000000 | 0.000000 | 0.197917 |

The selected-family means at 100k are:

| Family | Variants | Effective alphabet | Entropy | Max residue fraction | Max run | CA-step bad | Peptide bad | CA clashes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alanine-reference | 4 | 5.2372 | 0.535604 | 0.533691 | 8.11198 | 0.000151 | 0.000000 | 0.166667 |
| Position/element | 4 | 5.3128 | 0.537327 | 0.513428 | 7.95052 | 0.000123 | 0.000287 | 0.179688 |

At 100k, position/element is still the best **architecture family**: its
selected arms have slightly higher mean effective alphabet and entropy than
the selected alanine-reference arms, with lower residue concentration and
shorter runs. The difference is not large, so alanine-reference remains a
credible challenger rather than a discarded family.

The best **balanced individual configuration** is
`position_element/sig05_uniform_atomsc`: it retains high sequence diversity
while matching the lowest clash rate in the 100k panel (`0.114583`). The best
**sequence-only configuration** is
`position_element/sig10_polar2_atomsc`, which leads all 100k variants in
effective alphabet, entropy, residue concentration, and run length, but has
`0.270833` clashes. Therefore the raw-diversity winner should be treated as a
follow-up candidate, not the default promotion.

The names in this 100k factorial are sequence-objective factors:
`sig05`/`sig10` means `aatype_sigma_max=0.5`/`1.0 A`, `uniform`/`polar2`
means polar-class weight `1`/`2`, and `atomsc`/`pairsc` means atom-coordinate/
pair-distance self-conditioning. This is distinct from the older standalone
**Window16 low-noise sequence-gate family**, whose 25k aggregate clash rate
was `5.529` and which remains a diagnostic-only result.

Secondary-structure proxies are broadly stable rather than decisive. Mean
helix fractions are approximately 0.75--0.80 and sheet fractions approximately
0.08--0.11 across these screens. Small shifts should not be treated as an
architectural win without matched checkpoints and chemistry-aware diagnostics.

## Campaign-by-campaign interpretation

### 1. Position/element features: mature promotion candidate

This family remains the strongest all-around result after the 100k continuation.
Its `sig05_uniform_atomsc` arm is the current promotion candidate: it combines
high sequence diversity with the lowest clash rate among the leading 100k arms.
The `sig10_polar2_atomsc` arm is better for sequence diversity alone, but does
not have the same geometry margin.

This recommendation is still based on proxy geometry. The promoted candidate
should next receive the full stereochemical/sequence-topology diagnostic panel.

### 2. Alanine-reference features: highest upside, insufficient duration

At 25k this family has the highest effective alphabet and entropy, the lowest
maximum run among the feature screens, and a reasonable residue concentration.
That is the strongest evidence that an explicit ideal-alanine Atom14 reference
can help preserve sequence variety early in training.

It is not yet the overall winner because it has half the updates of the
position/element run. Its CA-step and peptide proxy errors are also higher than
the 50k position/element result. The correct next test is a 50k continuation,
not a conclusion from the current 25k snapshot.

### 3. Low-noise sequence gate: diversity without geometry

The standalone Window16 gate produces good sequence statistics after the fixed
sampler is applied, but the high clash rate makes the result unsafe to promote.
This pattern is consistent with a gate that improves logits or sequence
concentration without adequately constraining coordinate denoising. It should
only return to the promotion pool if a chemistry-aware loss or a matched
geometry intervention reduces clashes without sacrificing entropy. The 100k
`sig05_uniform_atomsc` result should not be interpreted as evidence that this
older family was rescued: it is a different feature family and a different
matched continuation.

### 4. Window16 index features: sampler-sensitive but not yet leading

This family improved substantially relative to its old screen. The old outputs
had strong collapse signatures; after the sampler correction the effective
alphabet increased to 4.65, entropy to 0.498, and maximum runs shortened to
8.57. That improvement demonstrates that the original failure was at least
partly accumulated sampling error rather than a definitive model defect.

Even after correction, it trails the position/element and alanine-reference
families in sequence diversity and has higher clash rates than the leading
candidate. It is useful as a control showing that the sampler fix changes
architectural interpretation, but it is not the next promotion choice.

### 5. Information-flow variants: no clear gain

The information-flow family ran to the longer 50k checkpoint, yet its fixed
outputs remain comparatively concentrated: effective alphabet 4.24, entropy
0.464, maximum residue fraction 0.605, and maximum run 9.92. Geometry proxies
are acceptable, but the sequence statistics do not show a compensating benefit.

The sampler correction did not materially reorder this family, unlike the
Window16 and pair-route cases. That makes the negative conclusion more credible:
there is currently no evidence that these information-flow changes improve the
model under the shared sampling contract.

## Old versus fixed sampler

The clearest changes from the reruns are:

- Alanine-reference, 25k: effective alphabet `4.214 -> 5.794`, entropy
  `0.464 -> 0.569`.
- Position/element, 50k: effective alphabet `3.413 -> 5.185`, entropy
  `0.329 -> 0.534`, maximum run `29.7 -> 7.7`.
- Window16 index, comparing its available old 10k screen with fixed 25k:
  effective alphabet `2.316 -> 4.652`, entropy `0.267 -> 0.498`, CA clashes
  `3.55 -> 1.04`.
- Information-flow, 50k: effective alphabet `4.335 -> 4.242` and entropy
  `0.473 -> 0.464`, i.e. essentially unchanged within this screen.
- Low-noise sequence gate, 25k: effective alphabet `3.029 -> 4.973`, entropy
  `0.320 -> 0.520`, maximum run `24.3 -> 8.5`; clashes remained about `5.53`.

The Window16 comparison is not a matched checkpoint comparison (old 10k versus
fixed 25k), so its magnitude is descriptive rather than causal. The other
comparisons use the same nominal checkpoint and isolate the sampler change more
cleanly.

The practical conclusion is that several earlier “architecture failures” were
actually accumulated sampling failures. The fixed sampler does not make every
family good, however: information-flow remains weak on sequence statistics,
and the low-noise gate retains a real geometry problem.

## Limitations

1. **Unequal training duration across families.** The two leading families
   now have selected 100k results, while Window16 index, information-flow, and
   the standalone low-noise-gate family remain at 25k/50k. Those older family
   means are useful context, not matched mature-checkpoint controls.
2. **Aggregate rather than per-variant reporting.** The means identify family
   behavior but can hide a strong or weak individual arm. Promotion should use
   the full per-variant manifests and factorial contrasts.
3. **Proxy geometry.** These screens report backbone/secondary-structure
   proxies and clashes. They do not replace the full stereochemistry panel for
   side-chain bonds, bond angles, chirality, peptide planarity, and
   sequence-topology agreement.
4. **No total-loss ranking.** Objective weights differ across experiments;
   total loss is not a valid cross-family quality metric.
5. **Sampler-specific evidence.** The results establish behavior under the
   corrected sampler. They do not prove that the trained denoiser is correct
   at every noise level; one-step diagnostics are needed to separate intrinsic
   denoiser error from accumulated Euler error.

## Recommended next steps

1. Promote **position/element + `sig05_uniform_atomsc`** as the default
   candidate for the full stereochemical and sequence-topology evaluation.
2. Keep **position/element + `sig10_polar2_atomsc`** as a diversity-oriented
   challenger, with clash reduction as its decisive gate.
3. Keep the standalone **low-noise sequence-gate family** diagnostic-only;
   require a substantial clash reduction before promotion.
4. Do not spend another long run on the information-flow family unless a new
   mechanistic hypothesis predicts a specific geometry or sequence benefit.
5. Report main effects and interactions only after matching checkpoint,
   sampler, sample count, and filtering rules across families.

## Bottom line

The best current configuration is **position/element features +
`sig05_uniform_atomsc` at 100k**. It is not the old standalone low-noise-gate
family; it is a position/element architecture arm using the 0.5 A sequence-loss
gate, uniform class weighting, and atom-coordinate self-conditioning. If
sequence diversity is prioritized over geometry, `sig10_polar2_atomsc` is the
leader, but its higher clash rate prevents it from being the default promotion.
Alanine-reference remains the closest competing architecture family. The fixed
sampler changes the interpretation substantially: Window16 and some pair-route
failures were sampling artifacts, while the information-flow and standalone
low-noise geometry weaknesses remain credible model-level signals.
