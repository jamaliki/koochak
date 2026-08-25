# Low-noise inverse-folding training and refinement sampler (2026-08-25)

Status: implementation complete; five of six 400k trajectories and their
terminal sample panels completed; matched aggregate quality analysis incomplete.

## Decision summary

The inverse-folding objective was added because corrected ProteinMPNN rescoring
showed that many mature Kaveh backbones are sequence-saveable even when their
jointly generated sequence fails. At step scale 2.25, ProteinMPNN produced
110/128 independently successful redesigns and rescued 31/32 source backbones
with at least one success among four designs. The latter is an oracle-over-four
diagnostic, not an independent designability rate; see
[the canonical corrected rescore](esmfold_designability_rescore_20260824.md#proteinmpnn-backbone-rescue).

The implemented response is a mixture objective, not a replacement for
unconditional generation. One quarter of examples present an almost fixed
backbone, heavily corrupt C-beta and distal atoms, hide the sequence, and train
sequence and side-chain recovery. Ordinary examples retain the established
quadrature side-chain corruption.

The implementation and launch are reproducible, but the campaign does **not**
yet support a quality conclusion. Five arms reached 400k; the sixth failed while
writing a checkpoint because Lustre returned `EIO`. Its terminal samples and
the six-arm aggregate were consequently skipped.

## Implemented objective

For each training example,

\[
z \sim \operatorname{Bernoulli}(0.25).
\]

### Ordinary examples (`z = 0`)

- retain the normal EDM corruption;
- with probability 0.5 draw `e ~ Uniform(0, 5 A)`, otherwise use `e = 0`;
- apply `sigma_sc = sqrt(sigma_bb^2 + e^2)` beginning at the arm's ordinary
  delayed slot;
- retain the complete generative coordinate, sequence, lDDT, and distogram
  objectives.

This is the training distribution that had already been accepted; the extra
noise is a random value in `[0, 5] A` on active examples, not a fixed 5 A bump.

### Inverse-folding examples (`z = 1`)

\[
\sigma_{\mathrm{bb}} = 0.05\ \mathrm{A}, \qquad
\eta \sim \operatorname{LogUniform}(0.1, 5.0)\ \mathrm{A}, \qquad
\sigma_{\mathrm{sc}} = \sqrt{0.05^2 + \eta^2}.
\]

The actual masks and losses are:

| Component | Implemented behavior |
| --- | --- |
| N, CA, C, O (Atom14 slots 0:4) | Corrupt at 0.05 A; exclude from coordinate and smooth-lDDT query losses |
| C-beta and distal atoms (slots 4:14) | Corrupt at `sigma_sc`; retain coordinate and smooth-lDDT supervision |
| Amino-acid input | Remains unknown, as in unconditional training |
| Amino-acid target | Full sequence cross-entropy remains active |
| Distogram | Per-example weight is zero |
| EDM input scaling | Uses the 0.05 A backbone clock; per-slot atom sigmas carry the side-chain clock |

C-beta deliberately belongs to the side-chain branch. Holding it fixed would
leak glycine status and local side-chain stereochemistry. The implementation
uses `inverse_folding_first_sidechain_slot = 4` regardless of whether an arm's
ordinary delayed corruption begins at slot 4 or 5.

The loss implementation uses the existing `coordinate_loss_mask` and the same
mask as the smooth-lDDT query mask. It does not silently retain backbone
coordinate pressure through an auxiliary loss. Sequence CE remains global.

## Matching two-stage sampler

The terminal campaign evaluates both the ordinary sampler and a dedicated
inverse-folding refinement pass.

1. Generate a complete structure and sequence with the current sampler.
2. Preserve N/CA/C/O from that generated structure.
3. Start a new denoising pass with C-beta and distal atoms reinitialized at a
   side-chain sigma of 5 A and with fresh unknown categorical state.
4. Run 64 EDM steps from 5.0 to 0.003 A with `rho = 5`, `gamma = 0`, Euler
   integration, and constant step scale 1.25.
5. Replace only slots 4:14 and the sequence prediction; return the final `x0`
   prediction while keeping the backbone fixed.

This differs slightly from the original verbal sketch, which proposed stopping
the first sampler exactly at backbone sigma 0.05 and screening side-chain scales
1.0-1.25. The launched implementation completes ordinary generation first and
uses 1.25 for the refinement pass. Future comparisons must describe the code
path actually run rather than the sketch.

Rigid alignment in the second pass fits on backbone atoms but applies the
resulting transform to all Atom14 slots. This separation is essential: the old
helper used its fit mask as an output mask, zeroed distal self-conditioning,
and made every nonzero side-chain-noise cap appear catastrophic. The regression
and repair are recorded in
[the sampling and geometry lessons](recycling_sampling_and_geometry_lessons_20260825.md#quadrature-self-conditioning-alignment-bug).

## Six-arm campaign

- Workflow: `hk-l128-b256-inverse-folding-6x400k-b023fc1-v1`
- Immutable commit: `b023fc176190b91806d383a5d1451ae787b153b9`
- Implementation/launcher commit: `728bcda`
- Root:
  `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/l128-batch256-inverse-folding-6x400k/b023fc176190b91806d383a5d1451ae787b153b9/v1`
- Batch: 256 local and global, one H100, gradient accumulation 1
- Length buckets: 64, 96, 128; maximum length 128
- Training: 400k steps, checkpoint every 10k, LR 3e-4
- Sampling: EMA BF16, matched seed 20260821, 32 samples at each of lengths
  64/96/128, ordinary and inverse-refinement panels

| Arm | Optimizer | Ordinary delayed slot | SS/3Di prediction and recycling | Terminal state |
| --- | --- | ---: | --- | --- |
| `current_inverse` | Adam, WD 0 | 5 | No | 400k and both sample panels succeeded |
| `ss3di_recycling_inverse` | Adam, WD 0 | 5 | Yes | 400k and both sample panels succeeded |
| `muon_slot5_inverse` | Muon, WD 0.01 | 5 | No | 400k and both sample panels succeeded |
| `muon_slot5_ss3di_inverse` | Muon, WD 0.01 | 5 | Yes | 400k and both sample panels succeeded |
| `muon_slot4_inverse` | Muon, WD 0.01 | 4 | No | Failed late on checkpoint-write `EIO`; terminal samples skipped |
| `muon_slot4_ss3di_inverse` | Muon, WD 0.01 | 4 | Yes | 400k and both sample panels succeeded |

The failed `muon_slot4_inverse` job was
`job-06c9bd6b498f739ad4b6`. The traceback terminated in checkpoint saving with
`OSError: [Errno 5] Input/output error`; it is an infrastructure/storage
failure, not evidence of a nonfinite model. The terminal aggregate task
`job-d137eca67c5f17c3a119` was skipped because it required all six arms.

## What is and is not learned

### Established

- The mixture corruption, side-chain-only coordinate/lDDT masks, full sequence
  CE, and distogram suppression are implemented and tested.
- The matching two-stage sampler freezes the backbone and reinitializes C-beta
  plus distal atoms without erasing distal coordinate self-conditioning.
- Five distinct batch-256 trajectories reached 400k and produced both terminal
  sample modes.

### Not established

- There is no valid six-arm ranking or aggregate designability result.
- The campaign does not isolate inverse-folding training from optimizer,
  ordinary delayed-slot, or SS/3Di effects; every arm has `q = 0.25`.
- The proposed `q = 0.5` arm and C-beta-fixed leakage control were not launched.
- The refinement pass has not been factorially compared at step scales 1.0 and
  1.25, nor against stopping the backbone sampler at sigma 0.05.

## Required completion

1. Recover the newest valid `muon_slot4_inverse` checkpoint or rerun only the
   missing tail, then generate its ordinary and inverse-refinement panels.
2. Run the aggregate analysis with corrected Atom37-masked ESMFold confidence;
   require `mean_plddt_source = pdb_atom37_masked_b_factors`.
3. Report ordinary versus refinement sampling within each arm before comparing
   arms. The sampler effect and training effect must not be conflated.
4. If the mixture is beneficial, run the missing causal controls: `q = 0`,
   `q = 0.25`, `q = 0.5`, and a C-beta-fixed leakage arm on a common optimizer
   and architecture.

## Provenance

- `728bcda`: implementation, tests, six-arm launcher, and refinement sampler.
- `b023fc1`: immutable allocation-fit campaign commit.
- `3e93bff`: corrected step-scale and ProteinMPNN saveability screen that
  motivated the mixture.
- Scruffy state inspected 2026-08-25: 21 succeeded, 1 failed, 91 blocked, and
  3 skipped tasks across the 116-task workflow.
