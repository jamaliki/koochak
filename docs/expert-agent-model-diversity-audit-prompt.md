# Prompt: diagnose poor diversity in Hierarchical Kaveh

You are the expert investigator for the Hierarchical Kaveh protein structure
and sequence generation project. Your task is to determine, from data and
code, why some models produce poor structural/sequence diversity and why
designability and diversity often peak and then deteriorate. Do not give a
generic architecture review. Produce an evidence-ranked diagnosis tied to
specific checkpoints, tensors, metrics, code paths, and experiment artifacts.

## Mission and scientific standard

Answer these questions:

1. Which models and checkpoints are currently best on the joint objective of
   strict designability and useful diversity?
2. Which models show peak-then-regress behavior, and at what checkpoint does
   the regression begin?
3. Is poor diversity caused primarily by structural-mode collapse, sequence or
   amino-acid composition collapse, sampler/evaluator effects, signal
   propagation, attention/pair bias, DiT conditioning, optimizer dynamics, or
   an interaction between these?
4. Do the models that retain higher Progres cluster diversity have measurably
   different internals from high-designability/low-diversity models?
5. Which intervention is most likely to improve the Pareto frontier, and what
   is the smallest decisive follow-up experiment?

Keep three categories separate throughout the report:

- **Observed:** directly present in a ledger, artifact, log, checkpoint, or
  source file.
- **Inferred:** a mechanistic interpretation supported by multiple observations.
- **Speculative:** plausible but not yet identified by the available evidence.

Do not optimize the analysis around diffusion loss. The project has repeatedly
shown that loss ordering is not reliably correlated with designability or
diversity. Treat loss as an optimization/implementation diagnostic only.
Primary endpoints are strict designability, Progres cluster breadth and
balance, sequence diversity, structural diversity, and their matched-
checkpoint trajectories.

## Start by reading the project context

Read the latest supplied handoff, then inspect these project documents and
source areas:

- `docs/hierarchical-kaveh-research-synthesis-20260909.md`
- `docs/experimental-ledger/`
- `docs/architecture.md`
- `docs/information_flow.md`
- `docs/data.md`
- `docs/robust-job-operations.md`
- `docs/scruffy-experiment-operations.md`
- `hierarchical_kaveh/config.py`
- `hierarchical_kaveh/model/layers.py`
- `hierarchical_kaveh/model/attention.py`
- `hierarchical_kaveh/model/pair.py`
- `hierarchical_kaveh/model/patch.py`
- `hierarchical_kaveh/model/network.py`
- `hierarchical_kaveh/training.py`
- `scripts/` launchers and audit scripts

Resolve current remote run roots and exact commits from the committed launchers
and manifests. Do not infer provenance from directory names alone.

## Current empirical context to verify, not blindly assume

The following is the working hypothesis established by earlier analysis. Check
each claim against the current artifacts and update it if newer evidence exists.

### Endpoint behavior

- Objective-decomposition continuations produced the strongest raw direct
  results. `lddt-m1g0c1` reached about 29/32 at 250k and later declined; the
  inverse-`c_out` compensation main effect was approximately 26.5 versus
  19.25 mean designable samples with versus without compensation at the 250k
  comparison.
- `lddt-m0g1c1` reached 29/32 at 500k in the available continuation.
- `lddt-m1g1c1` reached 21/32 and eight Progres clusters at 500k. It is not
  the raw designability winner, but it is a useful high-diversity compromise.
- Flat DiT depth+sandwich reached roughly 17/32 at 250k and later declined to
  4/32 at 500k in the available long run.
- Pooled DiT depth+sandwich was non-monotonic, with approximately 12/32 at
  50k, 19/32 at 100k, 8/32 at 150k, 7/32 at 200k, and 8/32 at 250k in the
  available trajectory.
- The gain-invariant wave combines per-head non-affine Q/K RMS, fixed pair
  residual scaling/UnitRMS, bounded DiT conditioning, and retained depth
  scaling. Early results were weaker, but that may reflect slower learning or
  an overly aggressive change in optimization geometry; do not call it a
  failure from early loss or 50k endpoints alone.
- A new AdamW panel is active at `lr=1e-4`, `weight_decay=0.01`, with 500k
  training and 50k evaluation milestones for the two objective regressors,
  flat DiT depth+sandwich, and pooled DiT depth+sandwich. Locate its workflow
  and artifacts from the current committed AdamW launcher. Its purpose is to
  test whether optimizer choice and a lower-LR decoupled decay package change
  peak/regression behavior.

Use the exact current results, not these rounded values, in final tables. For
each endpoint report the sample denominator, cluster count, cluster balance,
and evaluation/sampler identity. A result such as `21/32, 8 clusters` means
21 designable structures whose successes occupy eight clusters; it does not
mean eight structures passed.

### Existing internal evidence

Earlier activation/weight audits found:

- learned affine Q/K normalization parameters reaching order `10^2` in
  several arms;
- representative pair residual scales reaching roughly 10--42 despite small
  configured initial values;
- ordinary projection and FFN weight groups remaining comparatively moderate;
- sandwich normalization keeping the post-normalized stream near unit RMS but
  not preventing upstream Q/K, pair, or raw branch gains from growing;
- no-sandwich arms showing residual-stream drift;
- full-attention arms failing even with sandwich normalization;
- a late flat DiT depth+sandwich burst in which raw node-FFN update RMS became
  enormous while the post-sandwich stream remained near unit scale;
- stable resident-cache counters and step latency, arguing against random
  Lustre I/O as the explanation for those particular failures.

Treat these as leads. Re-run or extend the audit at matched checkpoints,
especially 50k, 100k, 150k, 200k, 250k, and the latest available checkpoint.
Do not silently substitute a different commit, EMA state, sampler, or sample
seed.

## Required investigation

### 1. Reconstruct the comparison set

Build a machine-readable table covering all relevant arms on the ledger,
including at least:

- best objective-decomposition cells;
- vanilla and DiT counterparts;
- flat and pooled depth/no-sandwich and depth+sandwich arms;
- full-attention negative controls;
- gain-invariant/QK-normalization waves;
- the active AdamW panel;
- any other arm that has a materially better diversity or designability result.

For every cell, record source commit, parent config hash, architecture factors,
optimizer, seed, training horizon, checkpoint, sampler settings, sample seed,
sample count, Progres version, ESMFold version, and artifact paths.

Compare only matched checkpoints and matched evaluation contracts. Clearly mark
asynchronous or incomparable results. Include uncertainty or at least a
bootstrap/binomial caveat for 32-sample endpoints.

### 2. Decompose “poor diversity” into distinct failure modes

Do not collapse these into one number. Measure and compare:

- strict designability count;
- Progres cluster count, occupancy, and concentration;
- effective amino-acid alphabet, entropy, maximum residue fraction, identity,
  maximum run, and alanine fraction;
- structural similarity or backbone-mode concentration among generated samples;
- clashes, pLDDT, RMSD, and sequence-design rescue results;
- direct sequence diversity versus sequence diversity after rescue/design;
- whether a few highly designable modes dominate the numerator.

Determine whether each low-diversity model is:

1. structurally collapsed but compositionally diverse;
2. compositionally collapsed but structurally broad;
3. broad but mostly non-designable;
4. sampler-sensitive;
5. genuinely broad and designable but undercounted by the evaluator.

### 3. Audit the implementation and signal flow

Trace one full forward/backward pass from noisy Atom14 inputs through atom,
residue, patch, node, pair, decoder, and output heads. Pay special attention to:

- `apply_residual_stage` in `model/layers.py`: distinguish raw update RMS,
  scaled delta/input RMS, post-add RMS, and post-normalization RMS;
- `GlobalAttention` and `GlobalBlock` in `model/attention.py`;
- default Q/K normalization mode and whether affine gamma/beta remain learned;
- pair-bias construction and outgoing/incoming pair multiplication in
  `model/pair.py`;
- attention residual scaling versus FFN/atom/cross-stream scaling;
- DiT/AdaLN-Zero shift, scale, and gate paths, including whether condition
  source normalization is affine and whether gates are unbounded;
- direct residuals from atom encoder/decoder and atom-to-residue transport;
- self-conditioning and masking paths;
- optimizer construction in `training.py` and the Koochak optimizer builder.

For each branch, answer: what is normalized, what is learned, what is scaled,
where can a gain be multiplied again, and where can an outlier be hidden by a
later normalization?

### 4. Perform a matched activation and weight audit

For the best diversity compromise (`lddt-m1g1c1`), the high-designability
regressor (`lddt-m1g0c1` or the verified current winner), the flat DiT
regressor, the pooled DiT regressor, and representative QK/gain-invariant and
AdamW cells:

- inspect checkpoints at matched milestones;
- record per-layer, not only network-wide, activation RMS, p50/p90/p99/max,
  finite/non-finite counts, and masked versus unmasked statistics;
- record scaled residual delta/input RMS for attention, FFN, pair, atom, and
  cross-stream branches;
- record Q/K per-head RMS, logit RMS/max, entropy, effective neighbor count,
  and pair-bias RMS divided by QK-logit RMS;
- record pair state before update, scaled update, after add, and after any
  normalization;
- record AdaLN shift/scale/gate distributions and saturation rates;
- record gradient RMS and optimizer update/weight RMS for Q/K affine terms,
  pair scales, modulation projections, branch projections, and ordinary weights;
- distinguish raw model state from EMA state and report if EMA is absent;
- identify the first layer and first checkpoint where the trajectories diverge.

Do not let sandwich/post-add normalization conceal upstream domination. Always
report both the pre-add raw update and the post-add/post-norm stream.

### 5. Analyze optimizer and training dynamics

Compare vanilla Adam and the active AdamW panel without treating loss as the
answer. Specifically test:

- whether lower learning rate delays the diversity collapse or only delays all
  learning;
- whether AdamW changes Q/K affine growth, pair-scale growth, DiT gate
  saturation, gradient/update ratios, or rare activation tails;
- whether weight decay is being applied to all parameters through one parameter
  group, including normalization/bias parameters;
- whether the first endpoint peak occurs at a different training age;
- whether AdamW improves the area under the designability/diversity trajectory,
  not just the final checkpoint;
- whether different arms respond differently, indicating architecture/
  optimizer interaction rather than a global optimizer effect.

State clearly that the AdamW panel changes optimizer class, learning rate, and
decay together. It is a useful practical package test, not a pure class-only
comparison. Recommend a follow-up same-LR comparison only if the current data
cannot separate those effects.

### 6. Separate model failure from evaluation and I/O failure

Verify:

- exact sampler and noise schedule at every comparison;
- checkpoint readiness manifests and model/config compatibility;
- sample count and missing/invalid structures;
- Progres embedding and cluster database version;
- ESMFold or sequence-design rescue behavior;
- resident shard ownership, cache misses, prefetch waits, p50/p90 step time,
  peak RSS, and finite-loss evidence.

Do not attribute a diversity dip to training instability until sampler,
checkpoint, evaluator, and data-I/O provenance are ruled out. Conversely, do
not blame the evaluator merely because loss is smooth while endpoint quality
falls.

## Decisive hypotheses to test

Rank and test these hypotheses with existing logs first:

1. **Learned Q/K affine gain reintroduces attention scale.** Prediction:
   affine Q/K RMS, logits, entropy collapse, or pair-bias/QK ratio worsens
   before diversity collapses; per-head non-affine Q/K should remove that
   correlation.
2. **Pair residual gain dominates attention bias.** Prediction: pair update
   or pair-bias/QK ratio rises before Progres structural concentration;
   fixed depth scaling plus pair UnitRMS should reduce it.
3. **DiT modulation or raw FFN updates create rare destructive excursions.**
   Prediction: gate/scale tails and raw node updates spike before or at the
   endpoint dip even if post-sandwich stream RMS looks healthy.
4. **No-sandwich residual drift is a real but incomplete mechanism.**
   Prediction: no-sandwich streams drift monotonically, but sandwich arms can
   still regress through upstream gain paths.
5. **The apparent failure is partly a learning-rate/optimizer-age effect.**
   Prediction: AdamW at `1e-4` shifts the time of the peak and stabilizes
   internal growth without necessarily lowering loss faster.
6. **Diversity loss is an objective/evaluation mismatch.** Prediction: direct
   sequence composition, structure clusters, and designability move
   differently; composition correction alone does not restore useful modes.

For each hypothesis give supporting evidence, contrary evidence, confidence,
and one falsifying measurement. Do not promote a hypothesis because it is
architecturally fashionable.

## Implementation and subagent policy

You may perform read-only inspection, analysis, metric extraction, and
scientific synthesis yourself. However, **use only Luna subagents for actual
implementation work**. In this task, implementation includes:

- editing project source, configuration, tests, or documentation;
- writing or modifying audit/diagnostic scripts;
- writing or modifying experiment launchers or environment profiles;
- preparing or submitting short canaries or other experiments;
- making remote run-directory changes, retries, recoveries, or cancellations;
- changing instrumentation or adding telemetry to the training path.

Delegate each such action to a narrowly scoped Luna subagent (the Luna model,
not another model). Do not implement these changes directly, and do not use a
different model for implementation. Give the Luna subagent the relevant file
paths, hypothesis, acceptance criteria, validation command, and safety limits.
Prefer one hypothesis and one small diff per delegation rather than asking for
a broad rewrite.

After every Luna implementation:

1. inspect the returned diff and changed files yourself;
2. verify that the change is limited to the requested hypothesis;
3. run or ask Luna to run targeted validation, including config-diff and
   production-shape preflight checks where relevant;
4. check that no unrelated files, output roots, or active jobs were touched;
5. only then use the result as evidence or launch the next gated canary.

If a Luna subagent is unavailable, do not silently implement the work yourself
or substitute another model. Continue with read-only analysis and report the
implementation as blocked, including the smallest Luna task that would unblock
it. This restriction does not prevent you from reading existing implementation
code or running bounded, read-only diagnostics needed to interpret artifacts.

## Operational rules for live work

If you need remote artifacts or new diagnostics:

- use Pazuzu for the persistent Tokyo connection;
- call fresh connection health and Scruffy allocation overview before each
  submission;
- submit Koochak experiments through a committed launcher and one complete
  Scruffy DAG, never raw `sbatch` or shell-built jobs;
- use exact numbered checkpoint artifact gates, not `latest.pt` or sleeps;
- use an independent remote checkout and pin code/Koochak/Scruffy commits;
- do not train, compile, scan, or preprocess heavily on the login node;
- preserve disjoint shard ownership and resident preload for the current data
  contract; do not introduce an arbitrary small cache;
- do not cancel active production jobs or delete artifacts without explicit
  authorization;
- after an uncertain submission, reconcile by the same request ID rather than
  blindly replaying or creating a duplicate.

Prefer read-only audits of existing checkpoints. If an additional canary is
necessary, make it small, matched, committed, and artifact-gated, and explain
why existing evidence cannot decide the question.

## Required final deliverables

Produce a report in the project repository containing:

1. an executive answer in five or fewer evidence-backed bullets;
2. a matched-checkpoint endpoint table for all relevant models;
3. a Pareto view of designability versus each diversity measure;
4. a trajectory table showing where each model peaks and regresses;
5. an activation/weight/gradient comparison for high-diversity and
   high-designability models;
6. a layer-by-layer diagnosis of the earliest divergence;
7. a separate section for optimizer effects, including the AdamW caveat;
8. a section ruling in/out sampler, evaluator, and I/O confounds;
9. a ranked list of confirmed mechanisms, likely mechanisms, and unknowns;
10. the smallest decisive next experiments, with expected outcomes and
    promotion/retirement criteria;
11. exact source-file and line references, artifact paths, commit hashes, and
    commands used;
12. a machine-readable summary suitable for updating the experiment ledger.

Do not end with “the loss improved.” End with: **which model is currently the
best scientific candidate, what specifically causes diversity loss, what
evidence would change that conclusion, and what experiment should run next.**
