# Hierarchy recycling throughput probe

Status: complete

## Hypothesis

Pallatom-style cumulative coordinate recycling can be adapted to the native
resolutions of Hierarchical Kaveh without rerunning a full stage:

- each coarse block predicts one C-alpha center per four-residue patch and
  reinjects its pairwise RBF distance embedding into the compact pair state;
- each residue-decoder block predicts one C-alpha and amino-acid logits per
  residue, then returns both to the residue stream;
- each atom-decoder block predicts cumulative Atom14 coordinates and amino-acid
  logits, returning non-terminal predictions to the atom stream.

The heads and feedback projections are shared across blocks. Feedback
projections and coordinate/logit outputs start at zero, so enabling the paths
preserves the initialized baseline function.

## Experiment

- Branch: `codex/recycling-throughput-probe`
- Hardware: one H100 per arm in the active Tokyo Scruffy allocation
- Workload: real filtered length-128 training, batch 32, BF16, compiled and
  fused, with the production data pipeline and optimizer
- Timing: 220 completed steps, first 80 discarded, 140 timed rows
- Instrumentation: synchronized CUDA phase timing, including forward, backward,
  optimizer, and complete-step durations
- Arms: baseline, coarse only, residue only, atom only, and all three
- Guardrails: p50/p95 completed-step time and peak CUDA allocated/reserved
  memory
- W&B: disabled

This screen measures engineering cost only. It does not establish a quality
benefit. Attempts 1-5 measure recycling without intermediate supervision;
Attempt 6 adds the missing Pallatom-style intermediate coordinate and sequence
objective.

## Results

### Attempt 1 - insufficient host-memory request

All five jobs in workflow `hk-hierarchy-recycling-throughput-896065a-v1`
failed during Inductor compilation in the 128 GB host-memory cgroup. The
baseline failed with the same DataLoader-worker OOM signature as the recycling
arms, so this attempt contains no architectural throughput result. It does show
that 128 GB is insufficient for this compiled model plus the resident-shard
profile pipeline. Attempt 2 raises the explicit request to 240 GB, matching the
repository's other compiled H100 canaries; model code and workload are
otherwise unchanged.

### Attempt 2 - asynchronous timer (memory result only)

All five jobs in workflow `hk-hierarchy-recycling-throughput-5e12e9d-v1`
completed, but their step timer did not synchronize CUDA. The apparent p50
step times therefore measured host launch latency rather than completed GPU
work: Koochak scalarized the loss, and thus waited for the GPU, only after the
timed interval. These timings and throughputs are invalid for comparing arms.

Peak CUDA memory remains valid:

| Arm | Peak allocated (MiB) | Peak reserved (MiB) |
| --- | ---: | ---: |
| baseline | 9617.5 | 9890 |
| coarse | 9638.0 | 9918 |
| residue | 9729.6 | 10012 |
| atom | 9720.7 | 9992 |
| all | 9853.3 | 10142 |

All recycling modes together add 235.9 MiB (2.45%) peak allocated memory.
Attempt 3 enables Koochak's synchronized phase profiler before drawing any
throughput conclusion.

### Attempt 3 - synchronized phase timing

All five jobs in workflow `hk-hierarchy-recycling-throughput-4bbd756-v1`
succeeded. Each arm contains 140 timed rows after 80 warmup steps. The profiler
synchronizes CUDA at every phase boundary, so absolute times include profiler
synchronization overhead; comparisons use the same instrumentation throughout.

| Arm | p50 step (ms) | vs baseline | p95 step (ms) | p50 samples/s | Peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 130.92 | - | 153.25 | 244.42 | 9617.5 |
| coarse | 146.43 | +11.84% | 170.51 | 218.53 | 9638.0 |
| residue | 138.90 | +6.09% | 159.94 | 230.38 | 9729.6 |
| atom | 138.86 | +6.06% | 158.70 | 230.44 | 9720.7 |
| all | 157.06 | +19.96% | 175.07 | 203.74 | 9853.3 |

The all-mode p50 forward/loss phase is 91.32 ms versus 76.93 ms (+18.70%),
and backward is 45.44 ms versus 36.91 ms (+23.12%). Optimizer, gradient-clip,
and EMA medians are essentially unchanged. Memory is not the constraint.

This is a feasibility result rather than a promotion result: there is one run
per arm, and the baseline ran on `gpu-10` while the four recycling arms ran on
separate devices on `gpu-3`. More importantly, coarse and all emitted a Dynamo
`recompile_limit` warning in `PatchLayout.pack`; baseline did not. The next
screen therefore compares baseline, coarse, and all after consolidating the
two new heterogeneous coarse pack calls into one, skipping a discarded final
atom coordinate reconstruction, and removing a redundant sequence-probability
cast round trip.

## Throughput improvement targets

1. Remove the coarse `PatchLayout.pack` compiler fallback and remeasure before
   attributing the remaining coarse cost.
2. If coarse remains material, specialize the existing fused patch-pair RBF
   projection for one point per patch. This would fuse distance, 16-bin RBF,
   and `16 -> 64` projection instead of materializing the roughly 2 MiB RBF
   tensor at each of eight coarse blocks.
3. For residue/atom feedback, benchmark folding the coordinate and sequence
   projections with `addmm`. At length 128, each separate narrow projection
   writes a full-width activation (about 6 MiB per residue block and 28 MiB per
   atom block). This candidate changes BF16 rounding and requires nonzero-weight
   forward/gradient parity plus a training-stability gate.

The supervised variant should be measured separately from architectural
recycling. Pallatom supervised the intermediate heads directly; relying only on
the final loss introduces delayed co-adaptation through zero-initialized
predictor and feedback projections.

### Attempt 4 - partial optimization, compile gate still failed

Workflow `hk-hierarchy-recycling-throughput-af4a44b-v1` repeated baseline,
coarse, and all with the same synchronized 220/80 protocol. Baseline and all
landed on the same physical GPUs as attempt 3; coarse used a different `gpu-3`
slot.

| Arm | p50 step (ms) | vs baseline | p95 step (ms) | p50 samples/s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 136.76 | - | 163.42 | 233.98 |
| coarse | 142.27 | +4.03% | 166.51 | 224.92 |
| all | 158.55 | +15.93% | 177.46 | 201.83 |

This attempt did not meet the compile gate: both coarse and all still emitted
`PatchLayout.pack`'s `recompile_limit` warning. Consolidating two new calls into
one reduced runtime gather work but still introduced one additional method
specialization. Attempt 5 inlines that single pure gather so no new
`PatchLayout.pack` specialization is created. Because the attempt-4 baseline
p50 was itself 4.46% slower than attempt 3, the change from +11.84% to +4.03%
coarse overhead is not treated as an attributable optimization win.

### Attempt 5 - compile-clean optimized result

Workflow `hk-hierarchy-recycling-throughput-768914b-v1` repeated baseline,
coarse, and all after inlining the one additional coarse gather. All three jobs
succeeded without a `recompile_limit` or graph-break warning. They used the
same physical GPU slots as their attempt-4 counterparts, so the before/after
comparison is better controlled than the first five-arm screen.

| Arm | p50 step (ms) | vs baseline | Mean step (ms) | p95 step (ms) | p50 samples/s | Peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 136.64 | - | 128.09 | 160.71 | 234.19 | 9617.5 |
| coarse | 140.84 | +3.08% | 130.78 | 159.12 | 227.20 | 9638.0 |
| all | 153.51 | +12.34% | 142.22 | 174.03 | 208.46 | 9853.3 |

Relative to attempt 4 on the same slots, baseline changed by -0.09%, coarse by
-1.01%, and all by -3.18%. The compile-clean all path's forward/loss phase is
11.50% slower than baseline and backward is 18.01% slower. This supports the
expected activation-traffic cost rather than an optimizer or data-pipeline
bottleneck.

### Attempt 6 - discounted intermediate supervision without Kabsch

Commit `0bf3ca0` retains every recycled block's postconditioned prediction and
adds the paper's discounted objective with `gamma = 0.99`, later blocks weighted
more heavily, and division by the total number of enabled blocks. Coarse blocks
are supervised against the masked mean clean C-alpha coordinate of each
four-residue patch. Residue blocks use clean per-residue C-alpha coordinates
and sequence CE; atom blocks use clean Atom14 coordinates and sequence CE. The
terminal atom block remains in both the intermediate and terminal objectives,
matching the paper's `k = K` term.

Per the experiment decision, intermediate coordinate targets are **not**
Kabsch-aligned. The established terminal coordinate objective retains its
configured alignment. Coordinate losses are stacked once per resolution and
all residue/atom sequence logits use one batched CE, rather than launching one
independent loss graph per block.

The first campaign ran nine synchronized 220/80 arms. Eight succeeded; the
all-unsupervised arm was excluded because nine-way cold Inductor compilation on
`gpu-6` stalled at the step-64 static bucket and reached its 3600-second job
limit before the measurement window. This is a compile-startup failure, not a
training-step result. The isolated stage pairs were:

| Stage | Unsupervised p50 (ms) | Supervised p50 (ms) | Supervision delta | Peak-memory delta (MiB) |
| --- | ---: | ---: | ---: | ---: |
| coarse, 8 blocks | 143.25 | 142.32 | -0.64% | 0.0 |
| residue, 3 blocks | 142.11 | 145.25 | +2.21% | +0.1 |
| atom, 3 blocks | 132.56 | 134.84 | +1.72% | +2.0 |

The sub-2.3% stage deltas are at the noise floor of one run: their mean and
tail movements are inconsistent with their p50 movements. They support a
bounded-cost conclusion, not precise stage attribution.

A script-only follow-up commit, `e471096`, repeated the all-stage pair alone
under workflow
`hk-hierarchy-recycling-throughput-e471096-warm-pair-v2`. Both jobs ran on
adjacent physical H100s on `gpu-3`, used identical data and self-conditioning
schedules, and completed without a Dynamo recompile-limit, graph-break, or
fallback warning.

| All 14 recycled blocks | p50 step (ms) | Mean (ms) | p95 (ms) | p99 (ms) | p50 samples/s | Peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unsupervised | 154.13 | 143.13 | 174.28 | 185.91 | 207.62 | 9853.3 |
| supervised | 160.05 | 149.34 | 192.66 | 240.43 | 199.93 | 9855.5 |
| relative delta | +3.85% | +4.34% | +10.55% | +29.32% | -3.70% | +2.1 MiB |

Matched-step analysis gives a less noisy view of the common case. Across the
same 140 post-warmup batches, supervised-minus-unsupervised complete-step time
has a +0.97 ms median (+0.64% median relative) and +1.79 ms 10%-trimmed mean.
The step-function portion adds +0.86 ms median and backward adds +0.51 ms
median. The +6.21 ms untrimmed mean is driven by a cluster of supervised
step-function stalls, including one 187 ms delta and repeated 40-84 ms deltas;
one repeat cannot distinguish a supervision-induced tail from per-GPU runtime
noise.

Relative to the attempt-6 baseline (137.49 ms p50), all-stage unsupervised
recycling is approximately +12.1% and the supervised variant is approximately
+16.4%. P50 throughput therefore moves from 232.75 samples/s at baseline to
207.62 without intermediate losses and 199.93 with them. Intermediate
supervision itself adds only 2.1 MiB to peak allocated memory; the complete
recycling-plus-supervision path adds about 238 MiB (2.47%) over baseline.

Cold compilation is the larger operational hazard for short jobs. Under
nine-way node-local contention, atom runs took 994-1616 seconds and one
all-stage run timed out; the isolated all-stage repeats completed in 562 and
579 seconds. Production launchers should precompile all static length and
self-conditioning variants or reuse a per-commit Inductor cache, and should
avoid placing many cold compilers on one node simultaneously.

## Conclusion

Hierarchy-native recycling with explicit intermediate supervision is
computationally feasible at length 128 and batch 32. All three loops plus all
14 discounted losses fit comfortably in H100 memory. Architectural recycling
is the main steady-state cost at roughly +12% p50; adding intermediate
supervision costs another +3.85% by aggregate p50, while matched steps indicate
the typical increment is closer to 1-2 ms. The latter distinction needs a
repeat before treating the observed p95 tail as causal.

The intermediate loss is already resolution-batched and does not merit a
custom kernel on this evidence. The next performance change should target the
larger recycling cost: benchmark a one-point specialization of the existing
fused patch-pair RBF projection for coarse recycling, then benchmark `addmm`
fusion of coordinate and sequence feedback for residue and atom recycling with
randomized nonzero weights and BF16 forward/gradient tolerances. Compile-cache
warmup should be fixed before another multi-arm screen.

No quality claim follows from this probe; it establishes only implementation
and resource feasibility. A quality canary can now train the explicitly
supervised intermediate predictions. Also decide whether atom recycling must be
load-compatible with an already-trained terminal head:
cumulative shared atom updates intentionally change that checkpoint's output
even when the newly added feedback matrices are zero.
