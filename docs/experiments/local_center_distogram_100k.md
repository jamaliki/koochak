# Local-center geometry and intermediate-distogram experiment

This current-main campaign compares two 100k-step arms:

| Arm | Pair initialization | Self-conditioned pair geometry | Intermediate prediction feedback |
| --- | --- | --- | --- |
| `local_center` | 16 ordered intra-patch C-alpha distances plus masked patch-center distances | yes, zero-initialized residual branch | no |
| `intermediate_local_center` | same | yes | shared intermediate distogram heads after each non-terminal coarse block; projected logits update the next pair state |

Both arms retain the existing residue-level C-alpha distogram target. The
intermediate arm additionally applies the same target after every non-terminal
coarse block with weight `0.25`. Its pair feedback projection starts at zero,
so the two arms have the same initial function before the learned recycling
path becomes active.

The legacy all-16-for-every-patch-pair geometry remains available as the
`legacy` mode and is used as the throughput canary baseline. The submission
workflow gates both candidates at p50 <= 8% and p95 <= 12% step-time
regression before training. It samples EMA checkpoints at 10k, 25k, 50k, and
100k using the fixed 200-step sampler panel at lengths 64/96/128.

## Results from the completed local-center screen

The 100k checkpoints were evaluated with 96 samples per arm (32 each at
lengths 64, 96, and 128), EMA weights, BF16, compilation enabled, and the
200-step sampler above. The final comparison is recorded at
`geometry-distogram-100k/fef3561/analysis/step100000/rescue_final_comparison.json`.

| Metric | `local_center` | `intermediate_local_center` |
| --- | ---: | ---: |
| Effective alphabet | **6.499** | 4.310 |
| Entropy (bits) | **2.640** | 2.046 |
| Maximum residue fraction | **0.423** | 0.597 |
| Maximum homopolymer run | **5.59** | 9.53 |
| C-alpha step bad fraction | 0.000 | 0.000 |
| C-alpha clashes per residue | 0.00263 | **0.00255** |

Intermediate feedback produced a strong early geometry improvement, but that
advantage largely disappeared by 100k while sequence concentration worsened:
the effective alphabet was 34% lower, entropy 22% lower, maximum residue
fraction 41% higher, and maximum homopolymer run 70% longer. Shared training
diagnostics at 100k were otherwise nearly identical (last-1000 averages for
coordinate loss 0.06643 vs 0.06597, terminal distogram loss 1.96768 vs
1.96748, and smooth-lDDT loss 0.35014 vs 0.34922). Intermediate feedback cost
approximately 6.8% mean step time and 4.1% p95 step time.

The geometry benefit also decayed over the trajectory:

| Step | `local_center` clashes/residue | `intermediate_local_center` clashes/residue |
| ---: | ---: | ---: |
| 10k | 0.01451 | 0.00925 |
| 25k | 0.01492 | **0.00345** |
| 50k | 0.00749 | 0.00651 |
| 100k | 0.00263 | 0.00255 |

The current conclusion is to retain `local_center` as the control and not
promote the current intermediate configuration. A follow-up recycling test
should reduce or anneal `loss.intermediate_distogram_weight` (currently 0.25)
only after the cross-patch feature has been evaluated.

## Cross-patch run status and continuation

The cross-patch arms were configured correctly (`max_steps: 100000`,
`ckpt_every: 5000`, explicit output directories, and no intermediate feedback),
but the first workflow stopped after writing only the 5k checkpoint. There was
no 25k/100k checkpoint or sampling manifest, so those arms were not completed
and were not included in the comparison above. The interruption was a workflow
termination rather than an intentional 5k limit.

Both arms have now been resumed from their valid 5k checkpoints:

- `cross_patch`: Scruffy job `job-93c1bce7db16bd311503`
- `cross_patch_extrema`: Scruffy job `job-e8357eb8b653a514e645`

They are running from
`/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/geometry-distogram-cross-100k/c248673`
with `--resume latest` and the existing 100k configs. At the restart check
both were active and had advanced to approximately step 8k. The next gate is
the matched 25k sample panel; only a cross-patch arm that preserves sequence
diversity there should be allowed to run to and be promoted from 100k.
