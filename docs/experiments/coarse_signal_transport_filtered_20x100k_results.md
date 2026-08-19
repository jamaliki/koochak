# Filtered/static-patching coarse signal-transport campaign: 20 x 100k results

## Executive decision

**Reject the proposed signal-stable parameterization.** Under the filtered
dataset and static group-of-four patching regime, combining nonzero internal
transport initialization with depth-scaled residuals made alanine collapse
substantially worse. At 100k, relative to matched legacy cells, the bundled
candidate reduced effective alphabet by `1.3817`, increased maximum-residue
fraction by `0.0974`, and increased alanine frequency by `10.45` percentage
points. All six depth-by-seed comparisons had the adverse direction on both
collapse metrics.

The effect is visible but not unanimous at 25k (five of six matched pairs) and
is unanimous at 50k and 100k. Thus 25k is an early warning, while 50k is the
first reliable rejection point for this intervention. The small improvement in
the all-atom nonbonded-clash proxy nearly disappears by 100k and does not
compensate for the sequence regression.

Keep `model.internal_transport_init: zero` and
`model.residual_parameterization: legacy` as the defaults. Do not add a
node-to-pair path in response to this result.

## Scope and provenance

The completed campaign has a fully crossed
`depth x transport_init x residual` factorial at two training seeds and a
third-seed rectangular confirmation of the bundled endpoint:

| Panel | Cells | Training seeds | Campaign digest | Manifest source commit |
|---|---:|---|---|---|
| Full factorial | 16 | `42`, `20260817` | `82b4dd3a2f5ea9e4d54ea7d838358a77f4dbc0257697b89382d54ea4294c18ba` | `ec9427d536bf0a91a20a3ec495ae44b14fc65088` |
| Bundled confirmation | 4 | `20260818` | `2e5ae11a6c0d170cfb10f92aaf5d9d13a2e16d8a704cbdd9373b731afa320875` | `ec9427d536bf0a91a20a3ec495ae44b14fc65088` |

The pinned Koochak commit is
`d186e7cc1533446165e5a92d504f2c5a0c051409`. Terminal sampling and analysis
were recovered from detached checkout `22629921add41f20525ae4c64f77d4bdff49a217`;
the recovery-only commits changed validation and launch control, not model or
data semantics.

The authoritative Tokyo roots are:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  coarse-signal-transport-filtered-ca4-82b4dd3a2f5e/
  coarse-signal-transport-filtered-ca4-seed3-2e5ae11a6c0d/
```

Every cell has 96 EMA samples at 25k, 50k, and 100k: 32 samples each at
lengths 64, 96, and 128. Sampling used seed `20260813`, BF16 compiled
inference, and the 200-step corrected sampler. The three full-factorial reports
contain 1,536 rows each; the three third-seed reports contain 384 each, for
5,760 evaluated structures in total.

## Dataset, topology, and model contract

The exact eligibility and topology contract was:

| Item | Applied semantics |
|---|---|
| Length | 32-128 inclusive |
| Mean pLDDT | strict `> 80` |
| Maximum loop length | strict `< 15` |
| Loop content | strict `< 0.4` |
| Packing density | strict `> 0.3` |
| CA continuity | offline exclusion if any same-chain, consecutively numbered CA pair exceeds 4 Angstrom |
| Eligible population | 23,727 samples |
| Patching | consecutive groups of at most four residues; restart on chain or residue-number discontinuity |
| Static patch capacities | `[16, 24, 32]` for the 64/96/128 buckets |

The base was the promoted local-center model with a six-layer atom decoder,
coordinate self-conditioning, no Kabsch alignment loss, no intermediate
distograms or feedback, and no node-to-pair path. Training used batch size 32,
BF16, `torch.compile` in `reduce-overhead` mode with `dynamic: false`, fused
backends, and no max-autotune.

This is not a dataset-only A/B against the predecessor campaign: the requested
patch-topology and compile-shape fixes also changed between campaigns. The
within-campaign factorial effects are identified; differences in magnitude
between the predecessor and this replication cannot be attributed solely to
the dataset filters.

## Completeness and validation

All six reports passed the strict campaign evaluator after fixing a validator
bug that incorrectly expected the embedded resolved-config digest to be null.
The corrected validator independently checked:

- the canonical campaign and resolved-config digests;
- every checkpoint filename, SHA-256, training seed, data seed, factor patch,
  and source/Koochak commit;
- the exact expanded cell set and 32 samples per cell-length;
- uniqueness of every `(cell, length, sample_index)` key; and
- the fixed sampling contract.

The main-effect estimates below were recomputed independently from cell
aggregates and matched the evaluator's row-level factorial contrasts to within
`1e-12`.

## Bundled candidate versus matched legacy

Values are `nonzero + depth_scaled` minus `zero + legacy`, averaged over the
six matched depth-by-training-seed pairs. Higher effective alphabet and entropy
are better; lower values are better for the remaining metrics.

| Step | Effective alphabet | Entropy (bits) | Max residue fraction | Alanine frequency | Max run | Nonbonded clashes/residue | Bad-bond fraction | Adverse pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25k | -0.8326 | -0.1874 | +0.0727 | +0.0791 | +0.9635 | -0.01893 | +0.00287 | 5/6 |
| 50k | **-1.8335** | **-0.4495** | **+0.1300** | **+0.1372** | **+2.5608** | -0.00881 | +0.00225 | 6/6 |
| 100k | **-1.3817** | **-0.3399** | **+0.0974** | **+0.1045** | **+1.6875** | -0.00121 | +0.00004 | 6/6 |

Alanine was the dominant amino acid in all 60 cell-milestone aggregate panels.
The intervention further concentrated that existing bias rather than changing
the identity of the collapsed residue.

At 100k, the matched effective-alphabet and maximum-fraction deltas were:

| Depth | Seed | Effective alphabet | Max residue fraction | Alanine frequency |
|---|---:|---:|---:|---:|
| 8 | 42 | -2.1776 | +0.1536 | +0.1845 |
| 8 | 20260817 | -1.0222 | +0.0695 | +0.0724 |
| 8 | 20260818 | -1.3880 | +0.1037 | +0.1053 |
| 12 | 42 | -1.2876 | +0.0999 | +0.1002 |
| 12 | 20260817 | -1.2407 | +0.0846 | +0.0827 |
| 12 | 20260818 | -1.1740 | +0.0729 | +0.0822 |

The rejection therefore does not depend on one depth, seed, or collapse
metric.

## Factor attribution

Main effects are high minus low from the fully crossed 16-cell, two-seed panel.
The third-seed endpoint panel is not pooled here because it cannot separately
identify residual and transport effects.

### Effective alphabet

| Factor effect | 25k | 50k | 100k |
|---|---:|---:|---:|
| depth: `c12 - c08` | -0.0216 | -0.5092 | -0.3411 |
| residual: `depth_scaled - legacy` | **-0.6871** | **-1.1077** | **-0.9879** |
| transport: `nonzero - zero` | -0.3499 | **-1.1455** | -0.4441 |

### Maximum residue fraction

| Factor effect | 25k | 50k | 100k |
|---|---:|---:|---:|
| depth: `c12 - c08` | +0.0011 | +0.0377 | +0.0246 |
| residual: `depth_scaled - legacy` | **+0.0474** | **+0.0707** | **+0.0664** |
| transport: `nonzero - zero` | +0.0396 | **+0.0843** | +0.0355 |

Depth-scaled residuals are the most persistent independent regression.
Nonzero transport is comparably harmful at 50k but weakens by 100k. At 100k,
the transport-by-residual interaction is also adverse (`-0.7090` effective
alphabet and `+0.0498` maximum fraction), so the bundle is worse than its
average main effects alone imply.

## Depth is not a universal causal explanation

The two-seed full factorial has a negative average depth effect at 50k and
100k, but the per-seed sign is not stable:

| Step | Seed 42 | Seed 20260817 | Seed 20260818 |
|---:|---:|---:|---:|
| 25k | +0.1221 | -0.1653 | +0.2480 |
| 50k | -0.6213 | -0.3971 | +0.2525 |
| 100k | -0.7784 | +0.0963 | +1.6584 |

Values are the `c12 - c08` effective-alphabet difference averaged over the
available signal variants within each seed. The 100k maximum-fraction depth
effects are `+0.0472`, `+0.0020`, and `-0.1206` in the same seed order.

Thus the campaign does not support a universal claim that 12 layers alone
cause collapse. It does support a stronger and more useful claim: the proposed
signal-stable bundle worsens collapse at both depths and across all three seeds
by 50k, even when the sign of the depth effect changes.

## Length and timing

The sequence penalty is largest at length 128 at every milestone:

| Step | Length | Effective alphabet | Max residue fraction |
|---:|---:|---:|---:|
| 25k | 64 | -0.6187 | +0.0566 |
| 25k | 96 | -0.6682 | +0.0677 |
| 25k | 128 | **-1.2111** | **+0.0938** |
| 50k | 64 | -1.6990 | +0.1176 |
| 50k | 96 | -1.7417 | +0.1221 |
| 50k | 128 | **-2.0596** | **+0.1503** |
| 100k | 64 | -1.2581 | +0.0803 |
| 100k | 96 | -1.3077 | +0.0953 |
| 100k | 128 | **-1.5793** | **+0.1165** |

The pooled 25k signal is large but has one matched exception. By 50k the
direction is unanimous and the magnitude is maximal; 100k establishes that
the failure persists rather than self-correcting.

## Geometry tradeoff

The candidate reduces the all-atom nonbonded-clash proxy by `0.0189`, `0.0088`,
and `0.0012` clashes/residue at 25k, 50k, and 100k. The benefit is much smaller
than in the predecessor campaign and is nearly absent at the terminal
checkpoint. The CA-step bad fraction is zero throughout, while the candidate
slightly increases the CA-clash proxy by `0.00078`, `0.00071`, and `0.00021`.

This remains evidence that the transport path changes learning, but not that it
fixes signal propagation. Its measurable sequence effect is large and adverse;
its terminal geometry benefit is negligible.

## Comparison with the predecessor campaign

The [predecessor ledger](coarse_signal_transport_20x100k_results.md) reached the
same reject decision. The bundled collapse effect is at least as severe in the
new regime by 50k:

| Step | Predecessor effective alphabet | New effective alphabet | Predecessor max fraction | New max fraction |
|---:|---:|---:|---:|---:|
| 25k | -1.1254 | -0.8326 | +0.0737 | +0.0727 |
| 50k | -1.0759 | **-1.8335** | +0.0799 | **+0.1300** |
| 100k | -0.5924 | **-1.3817** | +0.0530 | **+0.0974** |

Because filtering and patch topology changed together, this table establishes
robustness of the rejection, not a causal effect of the dataset filter.

## Throughput and performance contract

Compilation, fused-backend, and finite-gradient checks passed; every timing
summary contains 500 warmed rows and no max-autotune was used. Peak allocated
memory was about 10.46 GiB at depth 8 and 11.60-11.85 GiB at depth 12.

The final successful guards were rebaselined to the depth-12 legacy cell. At
fixed depth 12, the three main-panel signal variants had p50 ratios
`1.019-1.051` and p95 ratios `1.083-1.152`; the third-seed full bundle had
ratios `0.990` and `0.882`. Those fixed-depth comparisons pass the 1.20 limit.

They do **not** constitute a pass against the campaign's original depth-8
reference. Recomputed directly from the saved 500-row summaries:

| Depth-12 cell vs depth-8 legacy | p50 ratio | p95 ratio |
|---|---:|---:|
| Main legacy / zero | 1.233 | 1.147 |
| Main legacy / nonzero | 1.257 | 1.243 |
| Main depth-scaled / zero | 1.256 | 1.251 |
| Main depth-scaled / nonzero | 1.295 | 1.322 |
| Seed-3 legacy | 1.228 | 1.311 |
| Seed-3 full signal-stable | 1.216 | 1.157 |

Training proceeded under the explicit operational decision to start when the
ratios were close, not under a literal pass of the original depth-8 reference
contract. Future campaigns should revise the declared reference in the
campaign specification rather than rebaseline only the recovery guard.

## Execution and recovery ledger

- Workflows: `hk-coarse-signal-transport-filtered-ca4-v1` and
  `hk-coarse-signal-transport-filtered-ca4-seed3-v1`.
- Successful fixed-depth throughput guards: main
  `job-b54c2d8d776e5ad69efd` (attempt 6) and seed 3
  `job-18faaab69b39d7b0065a` (attempt 5).
- Initial 100k sampler tasks were dependency-skipped after earlier training
  attempts. Committed attempt-3 recovery `2262992` submitted only the 20
  missing terminal samplers and all six analyses; all 26 jobs succeeded.
- Main analysis jobs: 25k `job-e24d6855e86475d9d1b5`, 50k
  `job-70e15eb3e4bc5aa4f941`, and 100k
  `job-e386909de8e76ad292c8`.
- Seed-3 analysis jobs: 25k `job-790790352789a3a3df9e`, 50k
  `job-9d7e8b72ec64e536a69c`, and 100k
  `job-8274ba6acd6824298d34`.
- Provenance-validator fix `c4245a2` has 10 focused evaluator tests passing.

The six authoritative report files are
`analysis/milestone_step000025000.json`,
`analysis/milestone_step000050000.json`, and
`analysis/milestone_step000100000.json` under each run root.

## Decision

1. Retain zero transport initialization and legacy residuals.
2. Reject the bundled signal-stable parameterization; it worsens alanine
   collapse across depths, seeds, horizons, and sequence lengths.
3. Use 25k as an early-warning screen and 50k as the first decisive endpoint
   for architecture changes of this magnitude.
4. Do not treat coarse depth alone as the root cause; its effect is
   seed- and horizon-dependent.
5. Investigate the coupling between structural and sequence objectives rather
   than adding more internal transport. The intervention's small geometry gain
   comes with a much larger sequence-diversity loss.
6. Keep throughput reference changes explicit and versioned in future campaign
   specifications.
