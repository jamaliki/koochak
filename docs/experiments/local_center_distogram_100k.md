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
