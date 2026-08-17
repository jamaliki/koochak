# Local-center coordinate alignment control: no-Kabsch 100k results

## Scope and provenance

This entry records the matched coordinate-loss control for the local-center
3/3/8/3/3 architecture. The no-Kabsch arm changes only
`loss.align_coordinate_loss` from the aligned baseline; the model, sampler,
EMA weights, precision, sequence lengths, and sample count are otherwise
matched. Each milestone contains 96 EMA samples: 32 each at lengths 64, 96,
and 128.

The authoritative no-Kabsch aggregates are:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  local-center-coordinate-alignment-1x100k/
  88306f9a1eff48343fa5017aba76298bdba4b6f6/
  analysis/step050000-recovery.json
  analysis/step100000.json
```

The no-Kabsch run used sampling seed `20260817`, BF16 compiled inference,
EMA weights, and lengths `64/96/128`. The Kabsch column is the matched
aligned control from the same 3/3/8/3/3, 100k sampling contract.

## Matched comparison

Higher effective alphabet and entropy are preferable. Lower maximum residue
fraction, homopolymer run, and C-alpha clash proxy are preferable. The clash
rate is a coarse geometry warning proxy, not a stereochemical validation.

| Metric | Kabsch control, 100k | No-Kabsch, 50k | No-Kabsch, 100k | No-Kabsch vs control at 100k |
| --- | ---: | ---: | ---: | ---: |
| Effective alphabet | 4.634 | 5.552 | **6.719** | **+45.0%** |
| Entropy (bits) | 2.130 | 2.419 | **2.690** | **+26.3%** |
| Max residue fraction | 0.567 | 0.498 | **0.421** | **-25.7%** |
| Max homopolymer run | 8.969 | 7.427 | **6.865** | **-23.5%** |
| CA-step bad fraction | 0.000 | 0.000 | 0.000 | 0.000 |
| CA clashes/residue | 0.003852 | 0.003581 | **0.001166** | **-69.7%** |

## Interpretation

The no-Kabsch arm is descriptively better on every reported proxy at 100k,
including the geometry warning proxy. Its 50k-to-100k trajectory also moves
consistently toward higher sequence diversity and fewer geometry warnings:

| Metric | No-Kabsch change, 50k to 100k |
| --- | ---: |
| Effective alphabet | +1.168 |
| Entropy (bits) | +0.272 |
| Max residue fraction | -0.077 |
| Max homopolymer run | -0.562 |
| CA clashes/residue | -0.002414 |

This is a strong screening result for removing Kabsch alignment from the
coordinate loss, but it is not yet a causal estimate with uncertainty: the
comparison is based on one training run per condition, without training-seed
replicates or bootstrap intervals.
