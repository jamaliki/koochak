# Local-center coarse continuation: 150k and 200k results

## Scope and provenance

This entry records matched 150,000-step and 200,000-step sampling panels for
the six-arm coarse-depth/intermediate-feedback continuation. Each panel
contains 96 EMA samples per arm: 32 each at lengths 64, 96, and 128. The
panels share BF16 inference, lengths `64/96/128`, and sampling seed `20260817`.

The 150k manifests point to the corresponding `step000150000.pt` checkpoint.
The 200k training runs wrote the verified checkpoint as `step0200000.pt`.

The authoritative 150k aggregate is:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  local-center-coarse-continuation-6x200k/
  62233ea12be98e9220ebf532c0422fcb58d4b521/v1/manual-sampling-150k/
  analysis/milestone_step150000.json
```

The six sampler jobs succeeded. The first analysis job used the wrong
16-arm analyzer and failed on its seed contract; the recovered analysis used
the generic panel analyzer and succeeded. No sampling result was regenerated.

## 150k panel

Higher effective alphabet and entropy, lower maximum residue fraction and
homopolymer run, and lower C-alpha clash rate are preferable. Clash rate is a
coarse geometry warning proxy, not a stereochemical validation.

| Variant | Effective alphabet | Entropy (bits) | Max residue fraction | Max run | CA-step bad | CA clashes/residue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_deeper20_intermediate_332033` | **7.562** | **2.887** | **0.298** | **4.146** | 0.000 | **0.000244** |
| `coarse_deeper20_332033` | 7.575 | 2.873 | 0.317 | 4.521 | 0.000 | 0.004856 |
| `coarse_deep_intermediate_331233` | 7.437 | 2.848 | 0.338 | 5.094 | 0.000 | 0.000353 |
| `coarse_deep_331233` | 5.900 | 2.506 | 0.459 | 7.125 | 0.000 | 0.000597 |
| `coarse_deeper16_331633` | 5.975 | 2.466 | 0.464 | 6.479 | 0.000 | 0.000515 |
| `coarse_deeper16_intermediate_331633` | 5.421 | 2.369 | 0.466 | 6.927 | 0.000 | **0.000244** |

## What improved

The original `coarse_deep_331233` control improved descriptively from its
fixed-sampler 100k panel: effective alphabet increased from `5.294` to
`5.900`, entropy from `2.340` to `2.506`, maximum residue fraction fell from
`0.497` to `0.459`, and maximum run shortened from `7.406` to `7.125`.
The clash proxy rose from `0.000271` to `0.000597`. This is not a paired
longitudinal estimate because the 100k and 150k panels used different random
sampling seeds, so the direction is encouraging but not conclusive.

At 150k, intermediate feedback is depth-dependent:

| Matched pair | Effective alphabet delta | Entropy delta | Max-residue delta | Clash delta |
| --- | ---: | ---: | ---: | ---: |
| `331233`: intermediate minus control | **+1.537** | **+0.342** | **-0.121** | **-0.000244** |
| `331633`: intermediate minus control | -0.555 | -0.097 | +0.002 | **-0.000271** |
| `332033`: intermediate minus control | -0.013 | **+0.015** | **-0.019** | **-0.004612** |

Thus, intermediate feedback is a clear win for the 3/3/12/3/3 arm, a
sequence-diversity regression for the 3/3/8/6/3 arm, and nearly neutral on
diversity but strongly beneficial for geometry in the 3/3/12/6/3 arm.
At 150k, the best current balanced candidate was
`coarse_deeper20_intermediate_332033`; the non-intermediate deeper-20 arm had
similar sequence metrics but an approximately 20x higher clash proxy.

## 200k panel

The authoritative 200k aggregate is:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  local-center-coarse-continuation-6x200k/
  62233ea12be98e9220ebf532c0422fcb58d4b521/v1/
  manual-sampling-200k-v2/analysis/milestone_step200000-recovery.json
```

| Variant | Effective alphabet | Entropy (bits) | Max residue fraction | Max run | CA-step bad | CA clashes/residue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_deep_intermediate_331233` | **7.660** | **2.885** | **0.332** | **4.729** | 0.000 | **0.000081** |
| `coarse_deeper20_intermediate_332033` | 6.880 | 2.718 | 0.360 | 4.844 | 0.000 | 0.000190 |
| `coarse_deeper20_332033` | 6.837 | 2.706 | 0.374 | 5.229 | 0.000 | 0.002794 |
| `coarse_deep_331233` | 5.744 | 2.459 | 0.460 | 6.573 | 0.000 | 0.000515 |
| `coarse_deeper16_intermediate_331633` | 5.729 | 2.424 | 0.460 | 7.240 | 0.000 | 0.000244 |
| `coarse_deeper16_331633` | 5.627 | 2.374 | 0.485 | 7.281 | 0.000 | 0.000434 |

All six variants have 96 samples and lengths `64/96/128`. The 200k panel was
submitted as workflow `hk-local-center-coarse-continuation-sample-200k-1fc3c39-v2`.
The `coarse_deeper16_intermediate_331633` sampler hit a node-level
`CUDA is not available to the workload` preflight failure on its first
placement (exit 78); the single-arm recovery workflow
`hk-local-center-coarse-continuation-sample-200k-recovery-7165d5e-v1` reran it
successfully on a fresh placement. No other arm was duplicated.

## 150k to 200k trajectory

These are descriptive checkpoint-horizon changes under the same fixed sampler
contract, not paired training-seed estimates. Higher effective alphabet and
entropy, and lower maximum residue fraction, homopolymer run, and clash proxy,
are preferable.

| Variant | EA delta | Entropy delta | Max-residue delta | Max-run delta | Clash delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `coarse_deep_331233` | -0.156 | -0.047 | +0.001 | -0.552 | -0.000082 |
| `coarse_deep_intermediate_331233` | **+0.223** | **+0.037** | **-0.006** | **-0.365** | **-0.000272** |
| `coarse_deeper16_331633` | -0.348 | -0.092 | +0.021 | +0.802 | -0.000081 |
| `coarse_deeper16_intermediate_331633` | +0.308 | +0.055 | -0.006 | +0.313 | 0.000000 |
| `coarse_deeper20_332033` | -0.738 | -0.168 | +0.057 | +0.708 | **-0.002062** |
| `coarse_deeper20_intermediate_332033` | -0.682 | -0.169 | +0.062 | +0.698 | -0.000054 |

The 200k result makes intermediate feedback depth-dependent in a more nuanced
way than the 150k snapshot. The 3/3/12/3/3 intermediate arm is the clear
winner and improves further with training. The 3/3/12/6/3 pair remains nearly
neutral on diversity, while intermediate feedback preserves its geometry
advantage. The deeper-20 pair retains strong diversity, but both arms lose
some sequence diversity from 150k to 200k; intermediate feedback still
reduces its clash proxy by approximately 15x relative to the non-intermediate
arm.

## Limitations and next checkpoint

These are fixed-panel sampling summaries with 96 samples per arm. The
aggregate artifacts do not include paired bootstrap intervals, and there is
no training-seed replication. The ranking should therefore be treated as a
screening result, not a training-seed claim. The current lead to carry
forward is `coarse_deep_intermediate_331233`; the deeper-20 intermediate arm
remains a useful high-capacity geometry-safe alternative.
