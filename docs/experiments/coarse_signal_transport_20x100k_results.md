# Coarse signal-transport 20-run campaign: 100k results

## Executive decision

**Reject the proposed signal-stable parameterization.** The combination of
nonzero internal transport initialization and depth-scaled residuals made
single-residue collapse worse, not better. Across three training seeds, two
coarse depths, and the 25k/50k/100k milestones, the bundled candidate had a
lower effective alphabet in 17 of 18 matched comparisons and a higher maximum
residue fraction in all 18. The direction was already clear at 25k.

The intervention did improve the nonbonded-clash proxy, especially through the
nonzero transport factor. This is a real tradeoff rather than wholesale
training failure: the parameterization appears to move capacity toward geometry
while further concentrating the amino-acid distribution. Neither experimental
factor should be promoted at these settings.

## Scope and provenance

The completed campaign consists of a full `depth x transport_init x residual`
factorial at two training seeds plus a rectangular third-seed confirmation of
the bundled endpoint:

| Panel | Cells | Training seeds | Campaign digest | Source commit |
|---|---:|---|---|---|
| Full factorial | 16 | `42`, `20260817` | `6444d5a4fd3ac084be3c3b13eb2e6e5ae2e7c09203cdadb62a0d67da6650f2b5` | `f74bc79febfe1bd0cc1d0a970db16598461c64a6` |
| Bundled confirmation | 4 | `20260818` | `9b741f91f67b0333244fcd3810f10887734b00dcbc24d88503371a6f903e72cf` | `befc3d1ce46131aa442181f47ed17263d287ab35` |

The authoritative Tokyo roots are:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
  coarse-signal-transport-training-6444d5a4fd3a/
  coarse-signal-transport-training-seed3-9b741f91f67b/
```

Each cell has 96 EMA samples at each milestone: 32 samples at lengths 64, 96,
and 128 using the same sampling seed (`20260813`), BF16, and compiled sampling.
The full-factorial reports contain 1,536 rows per milestone; the third-seed
reports contain 384. Fixed-panel bootstrap intervals do not quantify
training-seed uncertainty.

## Bundled candidate versus matched legacy

The table reports candidate minus matched legacy, averaged over six
depth-by-seed comparisons at each milestone. For effective alphabet, positive
is better. For maximum residue fraction, maximum homopolymer run, clashes, and
bad-bond fraction, negative is better.

| Step | Effective alphabet | Max residue fraction | Max homopolymer run | Nonbonded clashes/residue | Bad-bond fraction |
|---:|---:|---:|---:|---:|---:|
| 25k | **-1.1254** | **+0.0737** | +0.5243 | -0.1467 | +0.0050 |
| 50k | **-1.0759** | **+0.0799** | +1.1615 | -0.0645 | +0.0017 |
| 100k | **-0.5924** | **+0.0530** | +0.7031 | -0.0459 | -0.0007 |

At 100k the effective-alphabet deltas for
`nonzero + depth_scaled` versus `zero + legacy` were:

| Depth | Seed 42 | Seed 20260817 | Seed 20260818 |
|---|---:|---:|---:|
| 8 | -1.2857 | +0.0197 | -0.9771 |
| 12 | -0.1688 | -0.7646 | -0.3781 |

The one near-zero effective-alphabet exception (`depth=8`, seed `20260817`)
still increased maximum residue fraction by `+0.0109`. The collapse decision
therefore does not depend on one metric or one seed.

## Factor attribution

Main effects below are high minus low and are estimated from the fully crossed
16-cell, two-seed panel. They should not be pooled with the third-seed bundled
panel because that panel cannot separately identify transport and residual
effects.

### Effective alphabet

| Factor effect | 25k | 50k | 100k |
|---|---:|---:|---:|
| depth: `c12 - c08` | +0.0641 | -0.1355 | +0.2310 |
| residual: `depth_scaled - legacy` | -0.0647 | **-0.6637** | -0.2677 |
| transport: `nonzero - zero` | **-1.0398** | +0.0215 | -0.2821 |

### Maximum residue fraction

| Factor effect | 25k | 50k | 100k |
|---|---:|---:|---:|
| depth: `c12 - c08` | -0.0058 | +0.0022 | -0.0268 |
| residual: `depth_scaled - legacy` | -0.0007 | +0.0371 | +0.0244 |
| transport: `nonzero - zero` | **+0.0649** | +0.0113 | +0.0230 |

Nonzero transport is the clear early driver: at 25k it reduces effective
alphabet by `1.0398` and raises maximum residue fraction by `0.0649`. Around
50k the average diversity penalty shifts toward depth-scaled residuals. At
100k both factors remain negative on effective alphabet, with strong
interactions:

| 100k effective-alphabet interaction | Difference-in-differences |
|---|---:|
| depth x residual | +0.9390 |
| depth x transport | -0.7727 |
| transport x residual | -0.3292 |

Those interactions explain why individual-arm ranks move with depth and
horizon. They do not rescue the bundled intervention, whose matched direction
is consistent across the panel.

## What this says about depth and timing

The depth main effect is small and non-monotone: `+0.0641` at 25k, `-0.1355`
at 50k, and `+0.2310` at 100k on effective alphabet. Thus this campaign does
not reproduce a simple claim that 12 layers always collapse more than 8 layers
once initialization and residual factors are crossed. The earlier observation
remains motivation for a signal-flow audit, but depth alone is not the stable
causal factor here.

For screening, 25k is sufficient to reject a bundled intervention when the
effect is this large and directionally consistent. The 50k and 100k endpoints
remain necessary for mechanism attribution: factor main effects and
interactions change materially with horizon.

## Geometry tradeoff

For nonzero versus zero transport, the all-atom nonbonded-clash main effect was
`-0.2160`, `-0.0883`, and `-0.0666` clashes/residue at 25k, 50k, and 100k.
The corresponding 100k bad-bond effect was `-0.0010`. These improvements do not
offset the collapse regression, but they are useful mechanistic evidence: the
new path carries signal and changes learning; it simply changes the balance in
the wrong direction for the sequence objective.

## Operational throughput history

The original performance gate failed. In the representative two-seed preflight
the depth-8 baseline p50 was `0.1225 s`, while the four depth-12 p50 ratios were
`1.184`, `1.224`, `1.228`, and `1.280`. The third-seed depth-12 ratios were
`1.242` and `1.258`. Compilation, fused-backend checks, and finite-gradient
checks passed; peak allocated memory was approximately `12.7 GB` at depth 8
and `13.9--14.8 GB` at depth 12.

The latency distributions also contained compile outliers caused by
data-dependent residue and patch tensor shapes. Training was later run through
separate production workflows after the operational decision to proceed. This
does not invalidate the scientific panels, but the failed guard must not be
reported as a performance pass.

The successor code removes active-residue cropping, variable-token packed
attention, and batch-dependent patch capacities. Patches are now consecutive
groups of at most four resolved residues, restarting only on chain or residue
number discontinuity. The filtered replication retains the strict `<=1.20x`
p50 and p95 gate and uses no max-autotune.

## Decision and successor

1. Keep `model.internal_transport_init: zero` and
   `model.residual_parameterization: legacy` as defaults.
2. Do not add node-to-pair transport as a response to this result.
3. Treat 25k as an early rejection checkpoint, but retain 50k/100k for any
   candidate that is not clearly dominated.
4. Run the same 20-cell design on the stricter dataset eligibility contract:
   length 32--128, mean pLDDT greater than 80, maximum loop length less than
   15, loop content less than 0.4, packing density greater than 0.3, and no
   same-chain consecutively numbered CA pair above 4 Angstrom.
5. Release production training only if every depth-12 canary stays within
   `1.20x` of the depth-8 legacy baseline on both p50 and p95 warmed step time.
