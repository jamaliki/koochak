# Fixed-sampler feature and architecture screen

## Executive summary

This note records the fixed-sampler reruns of the shorter feature and
architecture experiments that followed the stereochemistry factorial. The
reruns use the same trained checkpoints as the original screens; only the
sampling implementation changed. The fixed sampler uses a coherent perturbed-
time EDM grid and the recurrent self-conditioning frame expected by the
denoiser. It removes a sampling-time artifact that had made some pair-route
and sequence outputs collapse.

The provisional ranking is:

1. **Position/element features, 50k**: best balanced candidate and the safest
   promotion choice.
2. **Alanine-reference features, 25k**: highest sequence diversity, but not a
   fair winner yet because it has half as many updates as the 50k candidate.
3. **Low-noise sequence gate, 25k**: sequence diversity is competitive, but
   clashes remain high.
4. **Window16 index features, 25k**: substantially improved after the sampler
   fix, but still behind the two feature candidates above.
5. **Information-flow variants, 50k**: little evidence of a useful gain over
   the corresponding baseline.

This ranking is deliberately provisional. The 25k and 50k results are not
matched training durations, and total training loss is not comparable between
objective variants. The next decisive experiment is to train the top two
families to the same checkpoint schedule and evaluate them with the same fixed
sampler and diagnostics.

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

Each sampling panel used EMA weights, BF16, compile/fused execution, lengths
64/96/128, 32 samples per length, batch size 32, seed `20260813`, 200
perturbed-time Euler steps, the matching recurrent self-conditioning route,
`gamma=0.2`, noise scale `1.003`, step scale `2.25`, and sequence temperature
`0.1`. The trained model, optimizer, data, and scientific settings were not
changed for these reruns.

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

Secondary-structure proxies are broadly stable rather than decisive. Mean
helix fractions are approximately 0.75--0.80 and sheet fractions approximately
0.08--0.11 across these screens. Small shifts should not be treated as an
architectural win without matched checkpoints and chemistry-aware diagnostics.

## Campaign-by-campaign interpretation

### 1. Position/element features: current promotion candidate

This 50k campaign is the strongest all-around result. It combines high sequence
diversity with the cleanest geometry among the candidates that have run to 50k.
The fixed sampler also removed the concern that its pair or feature route was
being unfairly penalized by accumulated sampling error.

The remaining question is whether its diversity is retained at 100k or whether
it converges toward the lower-entropy behavior seen in some other families.
This family should be the first candidate for a matched 100k continuation and
for the full stereochemical/sequence-topology diagnostic panel.

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

The gate produces good sequence statistics after the fixed sampler is applied,
but the high clash rate makes the result unsafe to promote. This pattern is
consistent with a gate that improves logits or sequence concentration without
adequately constraining coordinate denoising. It should only return to the
promotion pool if a chemistry-aware loss or a matched geometry intervention
reduces clashes without sacrificing entropy.

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

1. **Unequal training duration.** Three families are at 25k and two at 50k.
   Sequence diversity can change materially with further optimization.
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

1. Continue **position/element** and **alanine-reference** under the same
   training contract to at least 50k, then 100k if both remain finite.
2. Run the exact same fixed sampling and one-step diagnostic panel at matched
   checkpoints, including the full stereochemical energy and sequence-topology
   agreement metrics.
3. Keep **low-noise sequence gate** as a diagnostic candidate only; require a
   substantial clash reduction before promotion.
4. Do not spend another long run on the information-flow family unless a new
   mechanistic hypothesis predicts a specific geometry or sequence benefit.
5. Report main effects and interactions only after matching checkpoint,
   sampler, sample count, and filtering rules across families.

## Bottom line

The best current bet is **position/element features at 50k** because they offer
the strongest compromise between sequence diversity and geometric validity at
the longest available duration. **Alanine-reference features at 25k** are the
most interesting challenger and may overtake it after a matched continuation.
The fixed sampler changes the interpretation substantially: Window16 and some
pair-route failures were sampling artifacts, while the information-flow and
low-noise geometry weaknesses remain credible model-level signals.
