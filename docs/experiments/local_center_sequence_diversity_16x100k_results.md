# Local-center sequence-diversity 16-arm results

## Scope and provenance

This records the completed 16-arm local-center sequence-objective campaign at
50k and 100k optimizer steps. The authoritative analysis artifacts are:

- `local-center-seq-16x100k/8b67f487454be864f7e712c9034e2166842e0e33/manual-sampling/analysis/milestone_step050000.json`
- `local-center-seq-16x100k/8b67f487454be864f7e712c9034e2166842e0e33/manual-sampling/analysis/milestone_step100000.json`

Each panel contains 96 EMA samples: 32 sequences at each of lengths 64, 96,
and 128. Bootstrap intervals are paired by sample index and stratified by
length; they quantify fixed-panel sampling uncertainty, not training-seed
uncertainty.

All 16 arms used the same architecture and geometry path:

- model depths: `3 / 3 / 8 / 3 / 3` (atom encoder / residue encoder / coarse /
  residue decoder / atom decoder);
- `pair_geometry_mode: local_center`;
- self-conditioned geometry enabled;
- `intermediate_distograms: false`;
- `intermediate_distogram_feedback: false`;
- terminal distogram loss only.

Thus, the result does not support an intermediate-prediction explanation for
the best variants. The differences below are sequence-loss schedule, polar
weight, and marginal-JS weight effects.

## Aggregate results

Rows are sorted by the 100k effective alphabet. Lower is better for pairwise
identity, maximum residue fraction, maximum homopolymer run, and clashes per
residue; higher is better for effective alphabet and entropy.

| Variant | EA 50k | EA 100k | H 50k | H 100k | ID 100k | Max fraction 100k | Max run 100k | Clashes/res. 100k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lin05to10_polar2_nojs` | 6.0537 | **6.4558** | 2.5361 | **2.6293** | **0.2286** | **0.4105** | **6.00** | 0.000895 |
| `lin05to20_polar2_js005` | 5.9242 | 6.4382 | 2.4992 | 2.6132 | 0.2463 | 0.4297 | 6.31 | 0.000570 |
| `lin05to10_uniform_nojs` | 4.9526 | 6.3225 | 2.2345 | 2.5787 | 0.2578 | 0.4500 | 6.76 | 0.002469 |
| `hard10_polar2_js005` | 5.9251 | 6.1948 | 2.4993 | 2.5532 | 0.2560 | 0.4419 | 6.24 | 0.000597 |
| `hard05_polar2_nojs` | 5.6925 | 6.1934 | 2.4338 | 2.5745 | 0.2558 | 0.4395 | 6.34 | 0.001329 |
| `lin05to10_polar2_js005` | **6.5093** | 6.1006 | **2.6343** | 2.5489 | 0.2532 | 0.4443 | 6.48 | 0.002116 |
| `lin05to20_uniform_nojs` | 4.7440 | 6.0142 | 2.1900 | 2.5145 | 0.2676 | 0.4536 | 6.52 | 0.000434 |
| `hard05_polar2_js005` | 6.4897 | 5.9841 | 2.6538 | 2.5280 | 0.2581 | 0.4479 | 6.07 | 0.000705 |
| `hard10_polar2_nojs` | 5.7167 | 5.9181 | 2.4537 | 2.5012 | 0.2687 | 0.4607 | 6.60 | 0.000651 |
| `hard10_uniform_nojs` | 5.9127 | 5.2908 | 2.5161 | 2.3181 | 0.3230 | 0.5240 | 7.82 | 0.000922 |
| `lin05to20_uniform_js005` | 4.4116 | 5.1816 | 2.0989 | 2.2947 | 0.3176 | 0.5115 | 7.74 | 0.000895 |
| `hard10_uniform_js005` | 5.4883 | 5.1704 | 2.4084 | 2.3019 | 0.3205 | 0.5198 | 7.90 | 0.000841 |
| `lin05to10_uniform_js005` | 4.2909 | 4.8779 | 2.0471 | 2.2177 | 0.3385 | 0.5434 | 8.48 | 0.004259 |
| `lin05to20_polar2_nojs` | 4.5347 | 4.8551 | 2.0945 | 2.2081 | 0.3302 | 0.5241 | 7.78 | 0.000922 |
| `hard05_uniform_nojs` (control) | 5.1008 | 4.5124 | 2.2884 | 2.0859 | 0.3868 | 0.5896 | 10.20 | 0.003689 |
| `hard05_uniform_js005` | 4.7606 | **4.0345** | 2.1431 | **1.9316** | 0.4239 | 0.6195 | 10.72 | 0.003933 |

## Main findings

1. `lin05to10_polar2_nojs` is the best 100k arm on the primary diversity
   metrics: effective alphabet `6.4558`, entropy `2.6293` bits, pairwise
   identity `0.2286`, and maximum residue fraction `0.4105`.
2. Relative to `hard05_uniform_nojs`, its paired bootstrap differences are:
   effective alphabet `+1.9475` (95% CI `[+1.5639, +2.3464]`), entropy
   `+0.5440` bits (`[+0.4412, +0.6435]`), maximum residue fraction `-0.1787`
   (`[-0.2103, -0.1483]`), maximum homopolymer run `-4.2140`
   (`[-5.1667, -3.3021]`), and clashes/residue `-0.002800`
   (`[-0.004449, -0.001356]`).
3. `lin05to10_polar2_js005` led at 50k but fell to sixth at 100k. The
   no-JS version therefore has the stronger mature result for this schedule.
4. Extending the soft tail to `2.0 A` is not uniformly beneficial: the
   `polar2 + JS` arm is competitive, but the no-JS and uniform variants are
   substantially worse at 100k.
5. Polar weighting is generally beneficial, but its effect depends strongly on
   the schedule and JS term. Marginal JS is not a universal improvement; its
   best mature use here is paired with `lin05to20 + polar2`.
6. The best variants did not use intermediate prediction or intermediate
   distogram feedback. Those mechanisms were disabled for all 16 arms, so this
   campaign cannot estimate their causal effect.

Full per-length metrics, amino-acid frequencies, audit rows, and all paired
bootstrap intervals remain in the two JSON artifacts listed above.
