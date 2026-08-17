# Local-center secondary-structure screen: 100k results

## Scope and provenance

This entry records the four-arm SeedProteo-style secondary-structure (SS)
conditioning screen on the local-center 3/3/8/3/3 architecture. The arms
share the same training and sampling contract; they differ only in whether SS
conditioning, prediction, recycling, and the auxiliary SS loss are enabled.

The authoritative analysis artifact is:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  local-center-secondary-structure-4x100k/
  1e3746a009862dc9d02e22c0712ef1d5d9302d5f/
  manual-sampling-100k-newalloc-285371/analysis/milestone_step100000.json
```

Training used source commit `1e3746a009862dc9d02e22c0712ef1d5d9302d5f`.
Sampling was run in the new Scruffy allocation `285371`, workflow
`hk-local-center-secondary-structure-sample-100k-newalloc-285371-1815b5c-v1`,
with 96 EMA samples per arm: 32 each at lengths 64, 96, and 128. The sampler
used seed `20260817`, BF16 compiled inference, and `all_x` SS input.

## Variant contract

| Variant | SS data/conditioning | SS prediction | SS recycling | SS auxiliary loss |
| --- | --- | --- | --- | ---: |
| `baseline` | no | no | no | 0.0 |
| `ss_input` | yes | no | no | 0.0 |
| `ss_aux` | yes | yes | no | 0.1 |
| `ss_recurrent` | yes | yes | yes (`alpha=0.5`) | 0.1 |

## 100k panel results

Higher effective alphabet and entropy are preferable. Lower maximum residue
fraction, homopolymer run, and C-alpha clash proxy are preferable. The clash
rate is a coarse geometry warning proxy, not a stereochemical validation.

| Variant | Effective alphabet | Entropy (bits) | Max residue fraction | Max run | CA-step bad fraction | CA clashes/residue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 4.579 | 2.113 | 0.571 | 9.125 | 0.000 | 0.004259 |
| `ss_input` | 5.888 | 2.514 | 0.412 | 5.063 | 0.000 | 0.002441 |
| `ss_aux` | 6.016 | 2.561 | **0.367** | **4.271** | 0.000 | 0.002306 |
| `ss_recurrent` | **6.300** | **2.593** | 0.404 | 5.469 | 0.000 | **0.001248** |

## What we learn

- Adding SS input conditioning alone improves every reported diversity and
  geometry proxy over the baseline.
- The auxiliary SS-prediction arm is the best concentration-control arm in
  this panel: it has the lowest maximum residue fraction and shortest maximum
  homopolymer run.
- The recurrent prediction arm has the strongest aggregate diversity and the
  lowest clash proxy. It is the best overall screening result, but the margin
  is descriptive rather than statistically established.
- All four arms have zero CA-step bad fraction at 100k. The observed geometry
  differences are therefore in the clash proxy, not in this bond-length
  screen.

## Checkpoint and uncertainty notes

The old 50k sampling attempts did not produce a valid panel because none of
the four training directories retained `step0050000.pt`; the sampler failures
were `FileNotFoundError`, not missing analysis output. This new-allocation
workflow intentionally samples only the verified `step0100000.pt` files and
has no checkpoint-waiting sampler dependency.

This is one training seed and one fixed 96-sample panel per arm. The table has
no training-seed replicates or bootstrap intervals, so small rank differences
should be treated as provisional. The aggregate values are weighted over the
64/96/128-length panel reported by the analyzer.
