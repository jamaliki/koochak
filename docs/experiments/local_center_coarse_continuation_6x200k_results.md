# Local-center coarse continuation: 150k results

## Scope and provenance

This entry records the matched 150,000-step sampling panel for the six-arm
coarse-depth/intermediate-feedback continuation. The panel contains 96 EMA
samples per arm: 32 each at lengths 64, 96, and 128. All six manifests point
to the corresponding `step000150000.pt` checkpoint and share BF16 inference,
lengths `64/96/128`, and sampling seed `20260817`.

The authoritative aggregate is:

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
diversity but strongly beneficial for geometry in the 3/3/12/6/3 arm. The
best current balanced candidate is therefore
`coarse_deeper20_intermediate_332033`; the non-intermediate deeper-20 arm has
similar sequence metrics but an approximately 20x higher clash proxy.

## Limitations and next checkpoint

These are fixed-panel sampling summaries with 96 samples per arm. The
aggregate artifact does not include paired bootstrap intervals, and there is
no matched 100k panel for all five non-control continuation arms in this
ledger. The ranking should therefore be treated as a screening result, not a
training-seed claim. The planned 200k samples are the next matched horizon;
the deeper-20 intermediate arm is the lead to carry forward, while the
deeper-20 non-intermediate arm remains a diversity control with a geometry
warning.
