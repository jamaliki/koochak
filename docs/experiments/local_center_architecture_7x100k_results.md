# Local-center architecture 7-arm screen: 100k results

## Executive summary

The first local-center architecture sweep completed its matched 100k sampling
panel for all seven variants. `decoder_double_33866` is the strongest diversity
variant at this checkpoint: it has the highest effective alphabet and entropy,
the lowest mean pairwise identity, the shortest maximum homopolymer run, and a
lower maximum-residue fraction than the reference. `wide_441044` is the second
strongest sequence-diversity candidate, but its CA-clash proxy is the highest
in the panel. `atom_shallow_23832` and `coarse_deep_331233` show a clear
diversity regression relative to the reference.

These are fixed-panel sampling results, not a claim about training-seed
uncertainty or downstream structural quality. The best next architecture
candidate is `decoder_double_33866`, subject to a repeat or longer-horizon
check with chemistry-aware validation.

## Architecture variants

All arms used the same data, optimizer, sequence objective, coordinate
objectives, self-conditioning, EMA, sampler, and 100,000-step horizon.
Intermediate distograms and intermediate distogram feedback were disabled.

| Variant | Atom encoder | Residue encoder | Coarse | Residue decoder | Atom decoder |
|---|---:|---:|---:|---:|---:|
| `ref_33833` | 3 | 3 | 8 | 3 | 3 |
| `atom_shallow_23832` | 2 | 3 | 8 | 3 | 2 |
| `residue_double_36863` | 3 | 6 | 8 | 6 | 3 |
| `coarse_deep_331233` | 3 | 3 | 12 | 3 | 3 |
| `wide_441044` | 4 | 4 | 10 | 4 | 4 |
| `encoder_double_66833` | 6 | 6 | 8 | 3 | 3 |
| `decoder_double_33866` | 3 | 3 | 8 | 6 | 6 |

## 100k sampling results

Values are means across lengths 64/96/128 and 32 samples per length. Higher
effective alphabet and entropy, lower maximum-residue fraction, lower mean
pairwise identity, and shorter maximum homopolymer runs indicate less sequence
collapse. CA clashes per residue is a geometry warning proxy, not a full
stereochemical assessment.

| Variant | Effective alphabet | Entropy (bits) | Max residue fraction | Max homopolymer run | Mean pairwise identity | CA clashes/residue |
|---|---:|---:|---:|---:|---:|---:|
| `decoder_double_33866` | **6.982** | **2.728** | **0.409** | **5.750** | **0.223** | 0.000353 |
| `wide_441044` | 6.644 | 2.656 | 0.415 | 6.188 | 0.234 | 0.003038 |
| `ref_33833` | 6.312 | 2.596 | 0.429 | 6.135 | 0.239 | 0.000353 |
| `encoder_double_66833` | 6.098 | 2.556 | 0.457 | 6.563 | 0.261 | 0.001845 |
| `residue_double_36863` | 6.005 | 2.508 | 0.460 | 6.729 | 0.267 | 0.002116 |
| `coarse_deep_331233` | 5.294 | 2.340 | 0.497 | 7.406 | 0.302 | **0.000271** |
| `atom_shallow_23832` | 5.229 | 2.325 | 0.506 | 7.729 | 0.309 | 0.002360 |

All seven variants had a unique-sequence fraction of 1.0 under this small
panel, so that metric does not discriminate the arms.

## Paired uncertainty against `ref_33833`

The analysis used 2,000 paired bootstrap draws stratified by length. Entries
are variant minus reference; intervals are bootstrap 95% intervals. The
intervals quantify fixed-panel sampling uncertainty only.

| Variant | Effective alphabet delta (95% interval) | Entropy delta (95% interval) | Max-residue-fraction delta (95% interval) |
|---|---:|---:|---:|
| `decoder_double_33866` | **+0.669** [+0.197, +1.123] | **+0.132** [+0.032, +0.236] | **-0.020** [-0.051, +0.012] |
| `wide_441044` | +0.332 [-0.099, +0.789] | +0.059 [-0.036, +0.156] | -0.015 [-0.047, +0.017] |
| `encoder_double_66833` | -0.215 [-0.572, +0.123] | -0.041 [-0.121, +0.039] | +0.027 [+0.001, +0.054] |
| `residue_double_36863` | -0.304 [-0.688, +0.060] | -0.089 [-0.180, -0.002] | +0.030 [-0.001, +0.061] |
| `coarse_deep_331233` | -1.013 [-1.378, -0.647] | -0.255 [-0.347, -0.156] | +0.068 [+0.038, +0.098] |
| `atom_shallow_23832` | -1.082 [-1.477, -0.691] | -0.270 [-0.366, -0.170] | +0.077 [+0.048, +0.105] |

The decoder-doubled arm is the only candidate with clearly positive paired
effective-alphabet and entropy intervals. The wide arm trends positive, but
its intervals include zero. The shallow-atom and coarse-deep regressions are
well separated from the reference on both diversity measures. The paired CA
clash differences are not a reliable ranking signal here: `wide_441044` has a
positive interval, while `coarse_deep_331233` is indistinguishable from the
reference despite having the lowest absolute clash proxy.

## Provenance and completeness

The canonical 100k artifacts were verified under:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  local-center-architecture-7x100k/4b271340acc5da5bfa6792918aec899fad0befb0/
```

The analysis artifact is `analysis/milestone_step100000.json` and points to
the canonical `samples/step100000` root. Its manifest contract is consistent
for all seven arms: EMA weights, BF16, compiled execution, seed `20260813`,
lengths 64/96/128, 32 samples per length, and checkpoint step 100000. Each
manifest points to its corresponding `train/<variant>/step0100000.pt`, and
all seven manifests and checkpoints were present. The analysis contains all
seven variant keys and uses `ref_33833` as its paired-bootstrap control.

Scruffy retained no matching jobs when checked after completion; the results
were therefore verified directly from the Sandpit artifacts and no jobs were
cancelled or resubmitted.
