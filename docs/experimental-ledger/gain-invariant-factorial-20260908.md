# Gain-invariant stability factorial (2026-09-08)

## Status

Active corrected launch. This factorial applies the gain-invariant package to
every active L128 architecture cell rather than only the current best DiT arm.

The first launch at `9955962` failed its preflights because the bounded DiT
path still dereferenced a removed MLP gate; it produced no scientific result.
The corrected launch at `691f07a` passed all 12/12 preflights and is currently
training 12 cells. Its downstream sample/ESMFold/analysis jobs remain blocked
on those training outputs, so no gain-invariant result is included yet.

## Scientific intervention

Each child run preserves its immutable parent architecture and changes only:

- `model.qk_norm_mode: per_head_rms`: parameter-free RMS normalization after
  reshaping Q and K into heads; fixed attention scale remains
  `head_dim**-0.5`.
- `model.pair_residual_mode: fixed_unit_rms`: fixed
  `1/sqrt(4*coarse_depth)` outgoing/incoming pair residual scale, followed by
  parameter-free UnitRMS after each pair add; learned pair scalars are removed.
- `model.block_conditioning_style: dit_bounded`: affine-free, condition-
  normalized bounded shift/scale modulation with no independent residual gate.

Depth-scaled residuals and existing parent architecture axes are retained.
Full-attention cells are included for diagnosis, not as production candidates.

## Coverage

The launcher covers the 12 active parent cells from the objective-decomposition
and transformer-stability factorials:

- 8 objective-decomposition cells at 50k;
- 4 transformer-stability cells at 500k;
- 50k-spaced sample, ESMFold, and Progres analysis milestones for each trainer.

The exact immutable parent hashes and resolved configuration diff are produced
by `scripts/submit_gain_invariant_factorial.py`.

## Diagnostics and gates

The production-shaped run must record, per layer where available:

- residual delta/input RMS and raw/post-add/post-norm RMS;
- per-head Q/K RMS;
- attention-logit RMS/max, entropy, and effective-neighbor count;
- pair state, scaled update, after-add, after-UnitRMS, and pair-bias/QK ratio;
- bounded DiT shift/scale distributions;
- gradient RMS and optimizer update/weight RMS for modulation parameters;
- cache hit/miss counts, prefetch waits, step latency, and finite-loss evidence.

Before treating the fused Q/K path as trusted, run the production-shaped BF16
forward/backward parity test in `tests/test_model_cuda.py` against native
PyTorch LayerNorm. The new per-head RMS path intentionally uses the native
portable implementation until a separate fused kernel is validated.

## Promotion rule

Promote only if the package improves or preserves the common structural quality
metrics while removing Q/K affine growth, pair-scale growth, attention-logit
collapse, non-finite values, and post-add drift. Reject variants that merely
bound final activations while raw branch or parameter gains remain unbounded.
