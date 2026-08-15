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

## Corrected cross-patch analysis at 25k

The two cross-patch arms were analyzed from the corrected 25k sample
directories only (96 samples per arm, 32 each at lengths 64/96/128). The older
non-suffixed sample directories are retained for provenance but are not used in
this comparison.

| Metric | `local_center` 25k | `cross_patch` 25k | `cross_patch_extrema` 25k |
| --- | ---: | ---: | ---: |
| Effective alphabet | 4.825 | **5.558** | 4.552 |
| Entropy (bits) | 2.205 | **2.390** | 2.098 |
| Maximum residue fraction | 0.550 | **0.506** | 0.534 |
| Maximum homopolymer run | 8.67 | 8.33 | **7.95** |
| C-alpha step bad fraction | 0.000 | 0.000 | 0.000 |
| C-alpha clashes per residue | 0.01492 | 0.01131 | **0.01004** |

At 25k, adding all cross-patch distances improves both diversity and geometry
relative to `local_center`: effective alphabet is 15% higher and clashes are
24% lower. Adding the max/min extrema feature lowers clashes by a further 11%
relative to `cross_patch`, but loses 18% of effective alphabet and 12% of
entropy. The plain `cross_patch` arm is therefore the better promotion
candidate unless the primary objective is geometry alone.

The old intermediate 25k sample panel is excluded from this table because it
was generated before the sampler recycling fix. The corrected intermediate
100k panel (non-compiled, with model-configured feedback active) measures
effective alphabet 4.190, entropy 1.995 bits, maximum residue fraction 0.611,
maximum homopolymer run 10.03, and C-alpha clashes per residue 0.00274.

## Cross-patch run status and continuation

The cross-patch arms were configured correctly (`max_steps: 100000`,
`ckpt_every: 5000`, explicit output directories, and no intermediate feedback).
The first workflow stopped after writing only the 5k checkpoint, and a 128 GB
resume hit a Slurm OOM kill in a DataLoader worker at about 10k. The 240 GB
resume subsequently wrote matched 25k checkpoints and continued producing
training log entries; the Scruffy terminal records are stale relative to the
remote run directory, so the checkpoint/log files are the authoritative status.

Verified 25k artifacts:

- `cross_patch/step000025000.pt`
- `cross_patch_extrema/step000025000.pt`
- 96 samples per arm under `samples/step025000/*_corrected` (32 each at
  lengths 64, 96, and 128)

The matched 25k sample manifests report compiled sampling. These arms do not
use intermediate prediction feedback, so compilation cannot disable recycling
in them. For the intermediate arm, the original compiled panel is not treated
as authoritative: a corrected non-compiled 100k panel was generated with
intermediate feedback enabled from the model config.

The resumed cross arms are:

- `cross_patch`: Scruffy job `job-c35f590fecdb2f9844d8`
- `cross_patch_extrema`: Scruffy job `job-ce87a4af8f09271c67b7`

They are running from
`/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/geometry-distogram-cross-100k/c248673`
with `--resume latest`, 240 GB per GPU, and the existing 100k configs. The
latest remote log check reached step 28,950 in both arms with approximately
0.123 s/step from the 25k-to-28.95k interval, implying roughly 2 h 25 min to
100k if that rate holds. Only a cross-patch arm that preserves sequence
diversity at the matched 25k panel should be promoted from 100k.
