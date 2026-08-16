# Local-center sequence-diversity 16-arm 100k implementation plan

## Status and objective

This document is an implementation handoff for a fresh agent. Implement and
launch a 16-arm sequence-objective campaign from scratch to 100,000 optimizer
steps. Keep the current `local_center` model, data, optimizer, coordinate
objectives, self-conditioning, EMA, and fixed sampler unchanged.

The campaign must answer three unresolved questions:

1. Does the earlier `1.0 A + 2x polar` gain come from the wider sequence-loss
   gate, polar weighting, or their interaction?
2. Can a soft sequence-loss tail above `0.5 A` retain the upside of wider
   supervision without reproducing the failure of the hard `2.0 A` gate?
3. Can a small batch-marginal composition loss reduce amino-acid collapse
   without degrading geometry or sequence-to-structure consistency?

Do not add intermediate distogram feedback, full cross-patch geometry,
alanine-reference features, or the older information-flow variants. Their
mature results do not justify including them in this expensive screen.

## Controlling evidence

The historical reference is `local_center` at 100k: effective alphabet
`6.499`, entropy `2.640 bits`, maximum residue fraction `0.423`, maximum
homopolymer run `5.59`, and C-alpha clashes per residue `0.00263`. The campaign
must include a newly trained exact control rather than compare new runs only to
that historical checkpoint.

The intervention choices come from two observations:

- `sig10_polar2_atomsc` improved effective alphabet by about 6% over
  `sig05_uniform_atomsc`, but changed the sigma gate and class weighting at the
  same time.
- A hard `2.0 A` gate reduced effective alphabet by 22% and worsened all other
  sequence and coarse-geometry metrics. A soft tail is therefore safer than
  another fully weighted high-noise gate.

See [the local-center results](local_center_distogram_100k.md) and
[the fixed-sampler feature screen](fixed_sampler_feature_architecture_screen.md).

## Fixed scientific contract

Copy [the winning config](../../configs/experiments/local_center_distogram_100k.yaml)
to `configs/experiments/local_center_sequence_diversity_16x100k.yaml`. Preserve:

- model depths `3 / 3 / 8 / 3 / 3`, patch size 4;
- `pair_geometry_mode: local_center`;
- `pair_self_conditioned_geometry: true`;
- `intermediate_distograms: false` and
  `intermediate_distogram_feedback: false`;
- maximum length 128, batch size 32, length buckets 64/96/128, seed 42;
- strict data filters `mean_plddt > 80` and `loop_content < 0.5`;
- coordinate weight 1.0, amino-acid weight 0.25, smooth-lDDT weight 1.0,
  terminal distogram weight 0.5, intermediate weight 0.0;
- Adam at `3e-4`, no weight decay, gradient clip 1.0;
- deterministic training self-conditioning probability 0.5;
- BF16, required compilation and fused kernels;
- constant EMA 0.999, updated every two optimizer steps with compensation;
- 100k steps and a 5k checkpoint cadence, which guarantees the required 10k,
  25k, 50k, and 100k checkpoints;
- the current 200-step sampler, EMA weights, seed `20260813`, lengths
  64/96/128, and 32 samples per length.

The base YAML must describe the exact control:

```yaml
loss:
  aatype_sigma_max: 0.5
  aatype_sigma_ramp_max: null
  polar_weight: 1.0
  aatype_marginal_js_weight: 0.0
```

Use a new W&B project `hierarchical-kaveh-local-center-seq-16x100k` and group
`local-center-seq-16x100k-v1`.

## The 16 variants

Cross four sequence-noise schedules with two polar weights and two marginal-JS
states. `js005` means a coefficient of 0.05 relative to the amino-acid CE,
before the shared outer `aatype_weight: 0.25`.

| Variant | Full CE through | Linear zero point | Polar weight | Marginal JS |
|---|---:|---:|---:|---:|
| `hard05_uniform_nojs` | 0.5 | none | 1.0 | 0.00 |
| `hard05_uniform_js005` | 0.5 | none | 1.0 | 0.05 |
| `hard05_polar2_nojs` | 0.5 | none | 2.0 | 0.00 |
| `hard05_polar2_js005` | 0.5 | none | 2.0 | 0.05 |
| `hard10_uniform_nojs` | 1.0 | none | 1.0 | 0.00 |
| `hard10_uniform_js005` | 1.0 | none | 1.0 | 0.05 |
| `hard10_polar2_nojs` | 1.0 | none | 2.0 | 0.00 |
| `hard10_polar2_js005` | 1.0 | none | 2.0 | 0.05 |
| `lin05to10_uniform_nojs` | 0.5 | 1.0 | 1.0 | 0.00 |
| `lin05to10_uniform_js005` | 0.5 | 1.0 | 1.0 | 0.05 |
| `lin05to10_polar2_nojs` | 0.5 | 1.0 | 2.0 | 0.00 |
| `lin05to10_polar2_js005` | 0.5 | 1.0 | 2.0 | 0.05 |
| `lin05to20_uniform_nojs` | 0.5 | 2.0 | 1.0 | 0.00 |
| `lin05to20_uniform_js005` | 0.5 | 2.0 | 1.0 | 0.05 |
| `lin05to20_polar2_nojs` | 0.5 | 2.0 | 2.0 | 0.00 |
| `lin05to20_polar2_js005` | 0.5 | 2.0 | 2.0 | 0.05 |

`hard05_uniform_nojs` must reproduce the existing loss function exactly. It is
the contemporaneous campaign control.

## Loss implementation

### Configuration

Edit `LossConfig` in `hierarchical_kaveh/config.py`:

```python
aatype_sigma_ramp_max: float | None = None
aatype_marginal_js_weight: float = 0.0
```

Validation requirements:

- `aatype_sigma_max > 0`;
- when set, `aatype_sigma_ramp_max > aatype_sigma_max`;
- `aatype_marginal_js_weight >= 0`.

The existing `aatype_sigma_max` remains the inclusive end of full-strength CE.
`null` must preserve the current hard-gate behavior.

### Sigma weights

Add a pure helper in `hierarchical_kaveh/diffusion/losses.py` that returns one
FP32 weight per sample:

```text
ramp_max is null:  w(sigma) = 1[sigma <= full_max]
ramp_max is set:   w(sigma) = clamp((ramp_max - sigma) /
                                    (ramp_max - full_max), 0, 1)
```

Thus the linear schedules have weight 1 at and below `0.5 A`, decay above
`0.5 A`, and reach exactly zero at 1.0 or 2.0 A. Keep this tensor calculation
compile-safe; do not branch on tensor values in Python.

Extend `aatype_cross_entropy` to accept per-sample floating weights while
preserving the current class-weighted, per-sample residue reduction. Average
the per-sample CE with the sigma weights. The hard-gate/no-JS path must match
the current result and gradients, including the differentiable-zero behavior
when every sample is inactive.

### Batch-marginal JS loss

Add a pure `aatype_marginal_js` helper. For valid residues in samples with
nonzero sigma weight:

1. Compute `q`, the sigma-weighted mean of `softmax(logits.float())` over the
   batch and residue dimensions.
2. Compute `p`, the sigma-weighted empirical target amino-acid distribution
   over the same residues. Do not apply polar class weights to `p`.
3. Return `0.5 * KL(p || m) + 0.5 * KL(q || m)`, where `m = 0.5 * (p + q)`.
4. Clamp probabilities only inside logarithms and retain a differentiable zero
   when no residues are active.

This campaign is single-GPU, so the loss may use the local batch marginal.
Document that a future DDP run would require a global differentiable reduction.

The total sequence contribution is:

```text
aatype_weight * (weighted_CE + aatype_marginal_js_weight * marginal_JS)
```

Do not replace CE with JS and do not maximize per-position entropy.
When `aatype_marginal_js_weight` is zero, take a config-static no-JS branch and
return a scalar zero for telemetry without computing softmax or JS. This keeps
the exact control's loss and runtime path unchanged.

### Telemetry

Return and log these metrics for every variant, including zero-valued controls:

- `aatype_loss` for weighted CE only;
- `aatype_marginal_js_loss` before its configured coefficient;
- `aatype_active_fraction`, the fraction with sigma weight greater than zero;
- `aatype_sigma_weight_mean`;
- existing coordinate, smooth-lDDT, distogram, and self-conditioning metrics.

Update `hierarchical_kaveh/training.py` so CSV, JSONL, stdout, and W&B receive
the new detached metrics.

## Campaign materialization and submission

Add `scripts/submit_local_center_sequence_diversity_16x100k.py`, following the
checkout, Koochak provenance, immutable-config, staging, and Scruffy patterns in
`scripts/submit_geometry_distogram_100k.py`.

The launcher must:

- define the variants as immutable dataclass values and generate exactly the
  16 names in the table;
- patch only the four experimental loss fields plus output/logging/W&B fields;
- assert each rendered config equals the base config plus its allowed patches;
- require a clean committed checkout, the pinned Koochak commit, and the
  independent Tokyo checkout named for the current commit;
- support `--dry-run` and `--stage-only` without submitting jobs;
- use stable, unique workflow/task/request IDs containing the variant and
  commit prefix;
- write beneath
  `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/local-center-seq-16x100k/<commit>/`;
- never overwrite another commit's outputs.

Create a two-step real-data H100 preflight for every rendered variant. Add two
250-step latency canaries, `hard05_uniform_nojs` and
`lin05to20_polar2_js005`, followed by the existing p50 <= 8% and p95 <= 12%
throughput guard. Production training may start only after all preflights and
the throughput guard succeed.

Submit 16 independent 100k training tasks. Use one H100, 14 CPUs, 240 GB RAM,
and a 48-hour limit per training task. Let Scruffy schedule them within the
active allocation; do not assume all 16 can run concurrently.

After each arm finishes, submit fixed-sampler jobs for its saved 10k, 25k,
50k, and 100k checkpoints. Each panel contains 96 EMA samples: 32 each at
lengths 64, 96, and 128, using the same seed and manifest contract as the
local-center campaign. Sampling must explicitly pass the configured recurrent
self-conditioning behavior through compiled inference.

Do not make 25k diversity rank an early-stop condition. Previous cross-patch
and alanine-reference rankings reversed between early and mature checkpoints.
Early samples are diagnostics; only nonfinite training, irrecoverable job
failure, or catastrophic geometry may stop an arm.

## Analysis implementation

Add `scripts/analyze_sequence_diversity_16x100k.py`. It must validate every
manifest before calculating results: checkpoint step, EMA weights, BF16,
compiled inference, lengths, 32 samples per length, seed, and the complete
sampling configuration.

Use the current definitions for effective alphabet, entropy bits, maximum
residue fraction, maximum homopolymer run, C-alpha step violations, and clashes
per residue. Add:

- unique-sequence fraction by length and its macro-average across lengths;
- mean pairwise sequence identity within each length only;
- aggregate amino-acid frequencies;
- deterministic paired bootstrap confidence intervals versus
  `hard05_uniform_nojs`, stratified by length and paired by sample index;
- milestone trajectories for every metric.

Write one JSON result per milestone and a final Markdown comparison under the
campaign's `analysis/` directory. Preserve per-sample rows or an equivalent
audit artifact so confidence intervals can be reproduced. Do not mix the old
normalized-entropy or raw-clash units into this campaign.

Report planned contrasts, not only the top cell:

- polar weight 2 versus 1 within every schedule/JS pair;
- JS 0.05 versus 0 within every schedule/polar pair;
- each schedule versus `hard05` within every polar/JS pair;
- schedule-by-polar, schedule-by-JS, and polar-by-JS interactions;
- 25k-to-100k rank reversals.

Bootstrap intervals measure sampling-panel uncertainty, not training-seed
uncertainty. State that limitation prominently.

## Tests and gates

Add or update tests for:

1. Config validation, strict loading, serialization, and backward-compatible
   defaults.
2. Exact hard-gate boundary behavior and equality with the existing baseline.
3. Linear weights below, at, between, and at/above both endpoints.
4. Weighted CE gradients: zero-weight samples have zero gradient; fractional
   samples contribute proportionally; all-zero batches are differentiable.
5. Marginal JS: zero for matched marginals, positive for collapsed predictions,
   finite with absent classes, mask-correct, and differentiable.
6. `compute_losses` total weighting and telemetry keys.
7. All 16 unique names and exact factor values.
8. Materialized configs differ only in allowed fields.
9. Workflow DAG, resource values, checkpoint naming, task counts, and unique
   request IDs.
10. Manifest rejection and deterministic analysis/bootstrapping on a synthetic
    mixed-length panel.

Run targeted tests first, then the full suite and lint using the repository's
micromamba environment, never system Python:

```bash
pytest -q tests/test_losses.py tests/test_config.py \
  tests/test_local_center_sequence_diversity_16x100k.py
pytest -q
ruff check .
python scripts/submit_local_center_sequence_diversity_16x100k.py --dry-run
```

Before production submission, inspect the dry-run JSON and verify exactly 104
tasks: 16 preflights, two latency canaries, one throughput guard, 16 training
tasks, 64 sampling tasks, four milestone analyses, and one final comparison.
No task may reference a path outside the current commit's output root.

## Promotion rule

Rank only the 100k panels. The primary endpoint is mean effective alphabet,
but a candidate is promotable only when all of the following hold against the
new `hard05_uniform_nojs` control:

- the paired-bootstrap 95% interval for effective-alphabet improvement excludes
  zero;
- entropy agrees in direction;
- maximum residue fraction and maximum homopolymer run do not regress;
- unique-sequence fraction does not fall and pairwise identity does not rise;
- C-alpha step violations remain effectively zero;
- clashes per residue increase by neither more than 25% relative nor more than
  0.001 absolute;
- the full stereochemical and sequence-to-structure panel finds no material
  regression.

If no cell passes every guardrail, keep `hard05_uniform_nojs`. If one or more
cells pass, rerun the control and the best candidate with a second training
seed before changing the project default. Do not infer training-seed
significance from the 96-sample bootstrap.

## Required deliverables

- [ ] New base experiment YAML.
- [ ] Backward-compatible sigma-weight and marginal-JS loss implementation.
- [ ] Training telemetry for the new loss terms.
- [ ] Immutable 16-arm launcher with preflight, throughput, training, sampling,
      and analysis DAG.
- [ ] Unified manifest-validating campaign analyzer and Markdown result writer.
- [ ] Targeted and full test-suite evidence.
- [ ] Dry-run task/config audit.
- [ ] Staged or submitted workflow receipt with commit and request IDs.
- [ ] Final 100k report containing planned contrasts, guardrails, uncertainty,
      and the second-seed recommendation.
