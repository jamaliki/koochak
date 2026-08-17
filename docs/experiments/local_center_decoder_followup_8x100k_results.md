# Local-center decoder follow-up: 50k and 100k results

## Scope and provenance

This entry records the eight-arm decoder follow-up panel at 50,000 and
100,000 training steps. Every arm has 96 EMA samples at each milestone: 32
each at lengths 64, 96, and 128. The final 100k panel uses the recovered
`coarse12_residue8_331283` arm and is complete across all eight variants.

Authoritative aggregates:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  local-center-decoder-followup-8x100k/
  1ecf2c66b659d0f0b335405e231dda5e4e7a40ae/v1/
  analysis/milestone_step050000.json
  analysis/milestone_step100000-recovery-v4.json
```

The fixed sampler used seed `20260813`, BF16 compiled inference, EMA
weights, and lengths `64/96/128`. The panel analyzer reported 8 variants x
96 samples at both milestones.

## Architecture variants

The three depth fields below are the coarse, residue-decoder, and atom-decoder
components represented by the arm names. The control is
`decoder_double_33866` (8/6/6).

| Variant | Coarse | Residue decoder | Atom decoder |
| --- | ---: | ---: | ---: |
| `decoder_double_33866` | 8 | 6 | 6 |
| `residue_decoder_double_33863` | 8 | 6 | 3 |
| `atom_decoder_double_33836` | 8 | 3 | 6 |
| `residue_deeper_33883` | 8 | 8 | 3 |
| `coarse12_decoder_double_331266` | 12 | 6 | 6 |
| `coarse12_residue6_331263` | 12 | 6 | 3 |
| `coarse12_residue8_331283` | 12 | 8 | 3 |
| `coarse12_residue8_decoder_double_331286` | 12 | 8 | 6 |

## Panel results

Higher effective alphabet and entropy are preferable. Lower maximum residue
fraction, homopolymer run, pairwise identity, and C-alpha clash proxy are
preferable. The clash rate is a coarse geometry warning proxy, not a
stereochemical validation.

| Variant | EA 50k | EA 100k | Entropy 100k | Max residue 100k | Max run 100k | Mean pair ID 100k | CA clashes/residue 100k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `residue_decoder_double_33863` | 5.009 | **7.216** | 2.800 | 0.380 | 5.563 | 0.204 | 0.002007 |
| `atom_decoder_double_33836` | 6.955 | 7.202 | 2.812 | 0.382 | **5.365** | 0.204 | **0.000163** |
| `decoder_double_33866` (control) | 5.811 | 6.974 | 2.727 | 0.410 | 5.792 | 0.225 | 0.000353 |
| `coarse12_residue8_331283` | 6.514 | 6.858 | 2.718 | 0.382 | 6.188 | 0.207 | 0.000271 |
| `coarse12_residue6_331263` | 6.143 | 6.808 | 2.712 | **0.374** | 5.292 | 0.209 | 0.001302 |
| `coarse12_residue8_decoder_double_331286` | **7.797** | 6.795 | 2.707 | 0.409 | 6.469 | 0.222 | 0.003120 |
| `residue_deeper_33883` | 5.403 | 6.099 | 2.551 | 0.449 | 6.510 | 0.260 | 0.000841 |
| `coarse12_decoder_double_331266` | 6.165 | 5.956 | 2.496 | 0.470 | 7.094 | 0.272 | 0.000597 |

All eight arms had a CA-step bad fraction of 0.000 and a unique-sample rate
of 1.000 at 100k.

## 100k uncertainty relative to control

These are paired bootstrap intervals for effective alphabet, stratified by
sequence length, using 2,000 draws with seed `20260813`. They quantify fixed
panel sampling uncertainty only; they do not include training-seed variation.

| Variant | EA delta vs control | 95% interval |
| --- | ---: | ---: |
| `atom_decoder_double_33836` | +0.229 | [-0.207, +0.665] |
| `coarse12_decoder_double_331266` | **-1.016** | **[-1.455, -0.564]** |
| `coarse12_residue6_331263` | -0.169 | [-0.625, +0.276] |
| `coarse12_residue8_331283` | -0.127 | [-0.633, +0.385] |
| `coarse12_residue8_decoder_double_331286` | -0.187 | [-0.724, +0.313] |
| `residue_decoder_double_33863` | +0.240 | [-0.203, +0.668] |
| `residue_deeper_33883` | **-0.875** | **[-1.313, -0.402]** |

## What we learn

- The best balanced 100k arms are `atom_decoder_double_33836` and
  `residue_decoder_double_33863`. The residue-decoder arm has the highest EA,
  while the atom-decoder arm has the lowest clash proxy and shortest maximum
  run.
- `coarse12_residue8_decoder_double_331286` is the 50k diversity winner but
  does not retain that lead at 100k: its EA falls to 6.795 and its clash proxy
  is the highest in the panel. This is a clear horizon-dependent rank
  reversal, not evidence that the 50k result was a stable winner.
- The control improves from EA 5.811 at 50k to 6.974 at 100k, so comparisons
  should be made at the same checkpoint rather than from the early ranking.
- `coarse12_decoder_double_331266` and `residue_deeper_33883` are the only
  arms whose 100k EA intervals exclude the control at the reported bootstrap
  level, both in the unfavorable direction.

These results support carrying the atom- and residue-decoder-doubled arms
forward, with a longer-horizon or replicated comparison before treating the
small lead over control as robust.
