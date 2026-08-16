# Coarse recurrent Atom14 decoder proposal

## Status

Proposed. This document defines an implementation and evaluation contract; it
does not change the promoted architecture until the throughput, correctness, and
scientific gates below pass.

## Decision summary

Replace the terminal full-width residue and atom decoder with eight shallow
recurrent units. Each unit runs one existing coarse block, decodes the coarse
state into a temporary 256-dimensional residue traversal, performs one local
Atom14 update, predicts a cumulative coordinate update, and feeds residue and
geometry deltas back into the persistent coarse single and pair streams.

The recurrent path must replace, not supplement, the current terminal decoder.
The 768-dimensional coarse single stream, 64-dimensional p=4 pair stream, and
existing outgoing/incoming triangle multiplications remain unchanged. Only the
temporary residue traversal is narrowed to 256 dimensions.

The first implementation uses all eight Atom14 updates. An anchor-only update on
alternating units is a follow-up performance ablation, not part of the initial
correctness target.

## Motivation

The current denoiser has one-way information flow:

```text
Atom14 encoder
  -> global residue encoder
  -> p=4 patchify
  -> coarse single/pair trunk
  -> unpatchify
  -> global residue decoder
  -> Atom14 decoder
  -> final outputs
```

This gives the coarse trunk no opportunity to consume geometry learned by the
fine decoder within the same denoiser call. Pallatom instead uses repeated
token/atom decoder units with temporary "traversing" atom representations and
coordinate-derived pair recycling. The published description is in
[Qu et al. 2025](https://proceedings.mlr.press/v267/qu25c.html); the released
eight-unit loop is visible in
[`alphafold/model/modules.py`](https://github.com/levinthal/Pallatom/blob/b27d70054dec6ce2f5ceadf8977de3d3baf00663/alphafold/model/modules.py#L1473-L1626).

A literal transplant would be too expensive here. Hierarchical Kaveh uses a
768-dimensional residue stream, whereas Pallatom's released token stream is 256
dimensional. It also introduces a p=4 scale transition that Pallatom's
residue-resolution pair stream does not pay. Repeating a full 768-dimensional
residue block and the existing Atom14 block eight times would make the model
roughly 1.8--2.2x slower. Narrow traversal states and replacement of the terminal
decoder reduce the forward-FLOP ratio to approximately 1.0--1.05x, leaving a
10--15% wall-time budget for recurrent memory traffic and launch overhead.

## Goals

1. Let predicted fine geometry update the coarse pair state within one denoiser
   call.
2. Let decoded residue and atom information update the next coarse single state.
3. Preserve the existing `DenoiserInput` and `Prediction` contracts.
4. Target median completed-step time within 1.10x of the matched broadcast
   baseline, with 1.15x as the hard rejection guardrail, at batch 32 and
   lengths 64, 96, and 128.
5. Preserve exact chain-safe p=4 layouts, partial tail patches, Atom14 masks,
   EDM preconditioning, and recurrent external self-conditioning.
6. Avoid repeated accumulation of the same residue broadcast in a persistent
   atom residual.
7. Keep the first implementation small enough to profile and reject cleanly.

## Non-goals

- Do not add triangle attention, pair transition, outer-product update, or a
  full AlphaFold 3 Pairformer.
- Do not add a persistent atom-pair tensor.
- Do not expand the coarse pair state to residue resolution.
- Do not add intermediate sequence, lDDT, or distogram losses in the initial
  implementation.
- Do not support dynamic recurrent depth inside one compiled graph.
- Do not migrate terminal-decoder checkpoints into the recurrent architecture.
- Do not use FP8, stop-gradient recycling, or alternating anchor-only units to
  make the initial throughput result pass.

## Fixed dimensions and notation

The proposed short-128 experiment fixes:

| symbol | meaning | value |
|---|---|---:|
| `B` | batch size per GPU | 32 |
| `N` | padded residue count | 64, 96, or 128 |
| `M` | chain-safe patch count | `ceil(segment_length / 4)` summed over segments |
| `R` | register count | 4 |
| `D_c` | coarse single width | 768 |
| `D_t` | traversal residue width | 256 |
| `D_a` | atom width | 128 |
| `D_p` | coarse pair width | 64 |
| `A` | Atom14 slots per residue | 14 |
| `K` | recurrent/coarse units | 8 |

Core tensors are:

| name | shape | lifetime |
|---|---|---|
| `residue_skip` | `[B,N,D_c]` | persistent encoder output |
| `atom_skip` | `[B,N,A,D_a]` | persistent encoder output |
| `coarse_x` | `[B,R+M,D_c]` | persistent and updated every unit |
| `pair` | `[B,M,M,D_p]` | persistent and updated every unit |
| `traversal_reference` | `[B,N,D_t]` | persistent projection of `residue_skip` |
| `traversal_residue` | `[B,N,D_t]` | scratch, rebuilt every unit |
| `traversal_atoms` | `[B,N,A,D_a]` | scratch, rebuilt every unit |
| `raw_update_sum` | `[B,N,A,3]` | persistent cumulative coordinate update |
| `predicted_coordinates` | `[B,N,A,3]` | scratch physical-coordinate estimate |

All residue and atom tensors remain prefix padded. `PatchLayout` remains the
single source of truth for multichain and discontinuous layouts.

## Architecture

### Encoder and initialization

The existing input, atom encoder, atom-to-residue pooling, global residue
encoder, patch layout, patchify, pair initializer, register tokens, and coarse
conditions remain unchanged through the point immediately before the current
coarse loop.

The recurrent decoder adds one persistent traversal reference:

```python
traversal_reference = residue_to_traversal(residue_skip)
```

`residue_to_traversal` is `RMSNorm(D_c)` followed by
`Linear(D_c, D_t, bias=False)`. It uses ordinary variance-preserving
initialization and is evaluated once per denoiser call.

Initialize:

```python
raw_update_sum = zeros([B, active_N, 14, 3])
last_atoms = atom_skip
```

The external self-conditioned coordinates continue to enter only through
`AtomInput`. They remain detached by `DenoiserInput.with_self_conditioning`.

### Static recurrent loop

The loop is statically unrolled over the existing `nn.ModuleList` of eight
unique coarse blocks. The traversal modules are shared across units.

```python
for unit_index, coarse_block in enumerate(self.coarse):
    coarse_x, pair = coarse_block(
        coarse_x,
        coarse_condition,
        pair,
        coarse_mask,
        coarse_positions,
        layout.pair_mask,
    )

    unit_condition = recurrent_condition(
        residue_condition,
        unit_embedding[unit_index],
        residue_mask,
    )

    traversal_residue = coarse_to_traversal(
        traversal_reference,
        coarse_x[:, REGISTER_COUNT:],
        layout,
        unit_index,
    )
    traversal_residue = traversal_residue_block(
        traversal_residue,
        unit_condition,
        residue_mask,
        token_positions[:, REGISTER_COUNT:],
    )

    traversal_atoms = traversal_atom_injector(
        atom_skip,
        traversal_residue[:, :active_N],
        active_atom_mask,
        unit_index,
    )
    traversal_atoms = traversal_atom_block(
        traversal_atoms,
        unit_condition[:, :active_N],
        active_atom_mask,
        segment_index,
    )

    raw_update_sum = raw_update_sum + coordinate_scale[unit_index] * coordinate_head(
        traversal_atoms,
        active_atom_mask,
    )
    predicted_coordinates = (
        c_skip[..., None] * raw_coordinates[:, :active_N]
        + c_out[..., None] * raw_update_sum
    )

    if unit_index + 1 < K:
        coarse_x = recurrent_single_feedback(
            coarse_x,
            traversal_reference,
            traversal_residue,
            atom_skip,
            traversal_atoms,
            layout,
            unit_index,
        )
    if unit_index + 1 < K or compute_distogram:
        pair = recurrent_geometry_feedback(
            pair,
            predicted_coordinates[..., 1, :],
            layout,
            unit_index,
        )

    last_atoms = traversal_atoms
```

The implementation should keep this loop in `CoarseRecurrentDecoder.forward`.
Do not put recurrent branches into `HierarchicalKaveh.forward` beyond input
assembly and output handling.

### Coarse-to-traversal expansion

`CoarseToTraversal` performs a learned ordered four-slot expansion:

1. normalize `[B,M,D_c]` coarse patch tokens;
2. apply `Linear(D_c, 4 * D_t, bias=False)` once per patch;
3. reshape to `[B,M,4,D_t]`;
4. multiply by `layout.slot_mask[..., None]`;
5. use `layout.unpack` to restore residue order;
6. add the result to `traversal_reference`.

The result is:

```text
traversal_reference
  + coarse_to_traversal_scale[k] * unpack(expand(coarse_patch))
```

The expansion must not first broadcast a 768-dimensional patch token to four
slots. Applying one `D_c -> 4D_t` GEMM before the view keeps the transition
contiguous and avoids a 768-wide intermediate.

For `layout.regular_contiguous`, `unpack` must reduce to a reshape/slice. The
existing gather path remains the correctness path for multichain inputs.

### Traversal residue block

The initial implementation uses one shared `GlobalBlock` configured as:

```text
width            = 256
condition_dim    = 256
heads            = 4
head_dim         = 64
ffn_expansion    = 4
dropout          = model.dropout
residual_scale   = 1 / sqrt(2 * K)
```

It operates on residues only, without learned registers. Global communication
already exists in `coarse_x`; this block restores residue-specific mixing after
ordered patch expansion. FA3 varlen attention remains the CUDA implementation,
using the original residue positions and mask.

Windowed traversal attention is not part of the first implementation. At
`D_t=256`, attention-score FLOPs are small relative to projections and the FFN,
so changing only the attention sparsity would not materially improve short-128
throughput.

### Recurrent condition

Shared traversal blocks require unit identity. Add a learned
`unit_embedding[K,D_condition]` to the ordinary diffusion condition:

```python
unit_condition = residue_condition + unit_embedding[k][None, None]
unit_condition = unit_condition * residue_mask[..., None]
```

Initialize `unit_embedding` from a zero-mean normal distribution with standard
deviation `0.02`. Do not append the unit embedding to the condition because that
would widen every conditioning projection.

### Traversal atom injection

Every unit reconstructs its atom state from the saved encoder state:

```text
traversal_atoms = atom_skip
  + atom_slot_embedding[k]
  + residue_projection(RMSNorm(traversal_residue))[:, :, None, :]
```

It must not consume the previous unit's `traversal_atoms`. This is the traversal
contract that prevents repeated coarse broadcasts from accumulating in a
persistent atom residual.

Use one shared slot embedding by default. Unit identity is already present in
`unit_condition`; per-unit slot embeddings add parameters without adding a new
signal.

### Traversal atom block

Use one shared Atom14 block with the existing four heads, head dimension 32,
residue-local radius-one attention, FFN expansion two, masks, and segment
boundaries.

The implementation must change how diffusion conditioning is applied. Current
atom blocks expand `[B,N,256]` conditions over 14 slots and concatenate the
expanded tensor with each atom. The recurrent block instead uses factored
projections:

```text
attention_gate = gate_atom(normalized_atoms)
               + gate_condition(unit_condition)[:, :, None, :]

ffn_scale = scale_atom(normalized_atoms)
          * broadcast(scale_condition(unit_condition))
```

The exact algebra need not match the current concatenated linear layer because
this is a new architecture trained from initialization. It must preserve:

- one condition projection per residue, not per atom;
- broadcast only after projection to `D_a`;
- zero-initialized output projections and bounded conditioning;
- the existing fused atom-window attention kernel.

This optimization is required to approach the 1.10x design target and pass the
1.15x throughput guardrail. It removes repeated condition GEMMs and large
`[B,N,14,D_condition]` concatenations.

### Cumulative coordinate prediction

One shared zero-initialized coordinate head maps normalized traversal atoms to
three raw update channels. Accumulate updates across units:

```python
raw_update_sum += coordinate_scale[k] * raw_update_k
```

`coordinate_scale` is a learned length-`K` vector initialized to `1 / sqrt(K)`.
The coordinate head is zero initialized, so the complete model retains the exact
EDM skip prediction at initialization.

After every unit, convert the cumulative raw update to physical coordinates
using the same `c_skip` and `c_out` tensors as the final output:

```python
predicted_coordinates_k = (
    c_skip[..., None] * raw_coordinates
    + c_out[..., None] * padded_raw_update_sum
)
```

Mask invalid atoms after combination. Pair refresh must consume physical
Angstrom coordinates, not `c_in`-scaled inputs or raw network updates.

### Fine-to-coarse single feedback

The next coarse block needs a patch-level summary of the current fine state.
Build it from deltas so the immutable encoder reference is not repeatedly added.
Apply this feedback after units zero through `K-2` only. The final unit has no
later coarse block, so a final single update would be dead work and would have
no task-loss gradient.

1. Compute `residue_delta = traversal_residue - traversal_reference`.
2. Compute a masked mean of `traversal_atoms - atom_skip` over Atom14 slots.
3. Project the atom mean from `D_a` to `D_t` and add it to `residue_delta`.
4. Pack the result to `[B,M,4,D_t]`.
5. Take a masked mean over the four slots.
6. Normalize and project `D_t -> D_c`.
7. Add the update only to the patch-token suffix of `coarse_x`.

Use a simple masked atom mean in the first implementation. The current learned
`AtomToResidue` pooling would repeat a comparatively expensive per-atom
`D_a -> D_t` projection and is not needed to establish the recurrent path.

The update is:

```text
coarse_patch += single_feedback_scale[k] * projected_patch_delta
```

`single_feedback_scale` is a learned length-`K-1` vector initialized to zero.
This makes the coarse execution stable at initialization while allowing an
immediate gradient to each used scale. The feedback projection receives
gradients after a scale moves away from zero; do not add a parallel always-on
compatibility path.

Register tokens persist unchanged through this feedback operation and continue
to change only inside the next `CoarseBlock`.

### Geometry-to-pair feedback

Use the four predicted C-alpha slots already defined by the p=4 layout. For each
patch pair, compute the same 16 ordered slot-to-slot distances and RBF embedding
used by `PairInitializer`, then project the geometry features to `D_p=64`.

The refresh excludes static separation, chain, and validity features because
those are already present in the persistent pair state. It computes only:

```text
geometry_delta[B,M,M,D_p]
```

and applies:

```text
pair = pair + geometry_feedback_scale[k] * geometry_delta
pair = pair * layout.pair_mask[..., None]
```

Use one shared geometry projection across units and learned per-unit scales
initialized to `0.05`. Unlike single feedback, a small nonzero geometry scale is
required so the recurrent pair path and its projection receive gradients on the
first step. The coordinate head remains zero initialized, so the initial
geometry refresh describes the EDM skip estimate rather than an arbitrary
network displacement.

The refresh is fully differentiable. Do not detach predicted coordinates inside
the denoiser. External previous-step self-conditioning remains detached exactly
as it is now.

The final unit refreshes `pair` only when `compute_distogram=True`, and the final
compact distogram is predicted from this refreshed state. This mirrors the
recurrent pair contract and gives the final geometry update a pair-loss
gradient. `self_condition`, which already requests no distogram, skips that
last refresh because no subsequent coarse block or public output consumes it.
The two static Boolean cases may compile as separate graphs. The scientific
screen must check whether the final refresh makes the distogram objective too
easy; it is not a reason to silently remove it.

### Final outputs

Remove the terminal `Unpatchify`, `residue_decoder`, `AtomOutput.inject`, and
`atom_decoder` calls in recurrent mode.

Return:

- coordinates from the final cumulative `raw_update_sum` and EDM combination;
- amino-acid logits from `last_atoms` using the existing final atom sequence
  head;
- the compact distogram from the final refreshed `pair`.

No intermediate tensors are added to `Prediction`. Optional debugging hooks may
record detached unit summaries, but training and sampling retain the current
public output contract.

## Parameter sharing and initialization

The eight coarse blocks remain unique. Share these modules across recurrent
units:

- `CoarseToTraversal`;
- 256-dimensional traversal residue block;
- traversal atom injector;
- optimized traversal atom block;
- coordinate head;
- atom-mean and patch-feedback projections;
- geometry refresh projection.

Keep unit-specific embeddings and scalar LayerScales. This gives each unit a
distinct operating point without repeating the traversal weights.

Residual-scale calculation is mode-specific and explicit. Terminal mode keeps
the current
`1 / sqrt(2 * (residue_encoder_depth + coarse_depth + residue_decoder_depth))`
calculation unchanged. Recurrent mode uses
`1 / sqrt(2 * (residue_encoder_depth + coarse_depth))` for the persistent
residue/coarse blocks and the separately specified `1 / sqrt(2 * K)` for the
shared traversal block. Do not retain a nominal terminal depth merely to tune
the recurrent scale.

Approximate new shared parameter counts are:

| component | parameters |
|---|---:|
| 256-wide traversal residue block | 1.25M |
| `768 -> 4*256` coarse expansion | 0.79M |
| reference and feedback projections | 0.39M |
| traversal atom block and adapters | 0.35M |
| geometry and coordinate heads | <0.05M |

The recurrent decoder is therefore roughly 2.8M shared parameters. Removing
the `3 x 10.4M` full-width residue decoder and three terminal atom blocks from
the profiled architecture reduces total parameter count by approximately 29M.
The exact count must be printed in preflight and recorded in the comparison
report.

Initialization contract:

| parameter | initialization |
|---|---|
| traversal reference projection | ordinary variance preserving |
| coarse-to-traversal expansion | ordinary variance preserving |
| traversal block residual outputs | existing zero-output initialization |
| atom injector projection | ordinary variance preserving |
| coordinate head | zero |
| coordinate scale | `1 / sqrt(K)` |
| single-feedback output | ordinary projection with `K-1` scales initialized to zero |
| geometry refresh projection | ordinary variance preserving |
| geometry feedback scale | `0.05` |
| unit embedding | normal, standard deviation `0.02` |

## Configuration contract

Add these `ModelConfig` fields:

```python
decoder_mode: str = "terminal"
traversal_dim: int = 256
traversal_heads: int = 4
traversal_head_dim: int = 64
share_traversal_weights: bool = True
recurrent_geometry_scale: float = 0.05
intermediate_atom_stride: int = 1
```

Validation rules:

1. `decoder_mode` is `terminal` or `coarse_recurrent` during the matched
   experiment. Delete `terminal` if recurrent decoding is later promoted as the
   only architecture.
2. `traversal_heads * traversal_head_dim == traversal_dim`.
3. `traversal_dim`, heads, head dimension, and `intermediate_atom_stride` are
   positive.
4. `intermediate_atom_stride == 1` for the initial implementation and production
   screen. Higher values are reserved for a later explicit ablation.
5. `coarse_recurrent` requires `coarse_depth > 0`.
6. `residue_decoder_depth == 0` and `atom_decoder_depth == 0` in recurrent mode.
   This prevents accidentally paying for both decoder paths.
7. The initial recurrent experiment requires `share_traversal_weights == True`.
8. `recurrent_geometry_scale` is finite and nonnegative.

The experiment configuration is:

```yaml
model:
  atom_encoder_depth: 3
  residue_encoder_depth: 3
  coarse_depth: 8
  residue_decoder_depth: 0
  atom_decoder_depth: 0
  decoder_mode: coarse_recurrent
  traversal_dim: 256
  traversal_heads: 4
  traversal_head_dim: 64
  share_traversal_weights: true
  recurrent_geometry_scale: 0.05
  intermediate_atom_stride: 1
```

Checkpoint metadata already contains the strict model configuration. Recurrent
checkpoints must fail to load against terminal configs and vice versa. Do not
write parameter-name or shape migration code.

## Source layout

### New file

Create `hierarchical_kaveh/model/recurrent.py` containing:

```text
RecurrentDecoderOutput
CoarseToTraversal
TraversalAtomInjector
TraversalAtomBlock
TraversalToCoarse
PairGeometryRefresh
CoarseRecurrentDecoder
```

`RecurrentDecoderOutput` is a frozen dataclass carrying only internal tensors:

```python
@dataclass(frozen=True)
class RecurrentDecoderOutput:
    coarse_x: Tensor
    pair: Tensor
    atoms: Tensor
    raw_update_sum: Tensor
```

`CoarseRecurrentDecoder` owns the eight coarse blocks and the shared traversal
modules. Its forward signature should receive already-built states and metadata;
it must not build `PatchLayout`, EDM coefficients, or model inputs internally.
`HierarchicalKaveh.forward` computes `c_skip` and `c_out` before invoking the
decoder and passes them with `raw_coordinates` and the static
`compute_distogram` flag.

Use a keyword-only interface so similarly shaped masks and conditions cannot be
silently exchanged:

```python
def forward(
    self,
    *,
    coarse_x: Tensor,
    coarse_condition: Tensor,
    pair: Tensor,
    coarse_mask: Tensor,
    coarse_positions: Tensor,
    residue_skip: Tensor,
    atom_skip: Tensor,
    residue_condition: Tensor,
    residue_mask: Tensor,
    residue_positions: Tensor,
    active_atom_mask: Tensor,
    segment_index: Tensor,
    raw_coordinates: Tensor,
    c_skip: Tensor,
    c_out: Tensor,
    layout: PatchLayout,
    compute_distogram: bool,
) -> RecurrentDecoderOutput:
    ...
```

`atom_skip`, `active_atom_mask`, and `segment_index` use the existing
`active_residues` prefix. Residue and coarse tensors retain their padded bucket
width. `raw_update_sum` is returned at the active width and padded once in
`HierarchicalKaveh.forward`; no recurrent unit allocates a full-width padded
copy.

### Existing files

- `hierarchical_kaveh/config.py`: add strict fields and cross-field validation.
- `hierarchical_kaveh/model/network.py`: construct the recurrent decoder, select
  one decoder mode, assemble its inputs, keep final EDM/output handling, and
  calculate the mode-specific residual scale without changing terminal mode.
- `hierarchical_kaveh/model/pair.py`: expose eager `pair_geometry_features` and
  `project_pair_geometry` functions used by both `PairInitializer` and
  `PairGeometryRefresh`. Keep `PairInitializer` parameter names stable in
  terminal mode. `PairGeometryRefresh` itself belongs to `recurrent.py`; do not
  duplicate RBF bounds or ordered slot-pair semantics there.
- `hierarchical_kaveh/model/kernels/patch_pair.py`: expose a geometry-only fused
  projection entry point usable after every unit.
- `hierarchical_kaveh/model/kernels/_patch_pair_rbf_triton.py`: support the
  recurrent forward/backward contract and the short-128 `M=16,24,32` shapes.
- `configs/experiments/coarse_recurrent_8x.yaml`: define the immutable recurrent
  experiment over the same data, loss, optimizer, and runtime contract as the
  matched baseline.
- `README.md`, `docs/architecture.md`, and `docs/pallatom_conformance.md`: update
  only after the architecture passes its promotion gate.

The canonical `PatchLayout.pack` and `unpack` methods remain in `patch.py`.
Avoid adding a second layout type or recurrent-only indexing convention.

## Kernel and compilation requirements

### Required for the first H100 throughput gate

1. Existing FA3 coarse pair-bias attention.
2. Existing fused pair-bias projection and triangle gate/residual kernels.
3. Existing fused Atom14 radius-one attention.
4. Factored per-residue atom conditioning without expanded condition
   concatenations.
5. Geometry refresh that does not materialize
   `[B,M,M,4,4,RBF_bins]` at the target CUDA shapes.
6. Static compilation of all eight recurrent units without a Python-dependent
   graph break between units.

### Implement only if profiling justifies it

- A fused coarse-to-traversal expansion/unpack kernel. The initial path should
  use one GEMM, reshape, mask, and the existing regular-layout view.
- A custom traversal-to-coarse reduction. Start with `layout.pack`, masked mean,
  and one GEMM.
- A fused complete recurrent unit. Existing FA3 and atom kernels already impose
  unavoidable boundaries, so a monolithic unit kernel is unlikely to be the
  first useful optimization.
- Compact pair-bias storage. The current dense bias is under 1 MiB at batch 32,
  `M=32`; it is not the short-128 priority.

### Compilation contract

- The number of units and all module calls are static for a model instance.
- `torch.compile(dynamic=False)` may compile once per existing length/patch
  bucket and self-conditioning input state.
- `require_compile` and `require_fused` remain fail-loud.
- Profile compile cache misses separately from warmed throughput.
- CUDA graphs or `mode="reduce-overhead"` are follow-up runtime experiments, not
  prerequisites for declaring the architecture correct.

## Gradient and activation-memory contract

The default recurrent graph is fully differentiable through:

- cumulative coordinate updates;
- predicted-coordinate RBF features;
- pair feedback and every later coarse block;
- atom-mean single feedback and every later coarse block.

Do not call `detach` inside `CoarseRecurrentDecoder`. The only detached geometry
is external previous-denoiser self-conditioning, as in the existing training
contract.

At short-128, run without activation checkpointing first. Narrow traversal
states and shared weights should fit the existing H100 envelope. If peak memory
exceeds the 1.15x gate, profile saved tensors before adding checkpointing.
Selective checkpoint order is:

1. traversal atom FFN;
2. pair triangle projections/contractions;
3. entire recurrent units only as a last resort.

Checkpointing all units recomputes most of one additional forward pass and is
expected to violate the 1.15x latency target.

Shared traversal parameters are reused eight times. DDP and compile tests must
verify that gradients accumulate exactly once per invocation and that reused
parameters participate in one all-reduce bucket without unused-parameter logic.

## FLOP and memory budget

The following are analytical dense-operation estimates, counting each
multiply-add as two FLOPs and excluding normalization, elementwise operations,
masks, and kernel padding. At `B=32`, `N=128`, BF16, the current operators are
approximately:

| operator | forward GFLOP |
|---|---:|
| 768-wide global residue block | 89.7 |
| current Atom14 block | 29.4 |
| current coarse block | 27.7 |
| 256-wide traversal residue block | 11.1 |

The profiled `3/3/8/3/3` baseline is approximately 965 GFLOP per forward. Its
post-coarse terminal decoder is approximately 363 GFLOP. Eight conservative
256-wide traversal residue + current atom + cross-scale units are approximately
398 GFLOP before the factored atom-conditioning saving. Replacing the terminal
decoder therefore gives an estimated whole-forward ratio of 1.04x. Factored atom
conditioning should recover another 50--60 GFLOP across eight units.

The design target is nevertheless 1.10x rather than the FLOP ratio because atom
blocks, geometry projection, masks, reductions, and 32x32 triangle batched
matmuls are expected to be bandwidth- or launch-bound on H100. The isolated
profile must confirm that attribution. A 1.15x p50 is the hard rejection
guardrail, not the desired steady-state result.

Record these peak allocated tensors during the isolated profile:

- pair state and triangle operands;
- traversal residue activations;
- Atom14 QKV and FFN activations;
- geometry-refresh saved tensors;
- full compiled model peak allocated/reserved memory.

No recurrent implementation should materialize an `[B,N,14,256]` condition or
an `[B,M,M,4,4,RBF_bins]` geometry tensor on CUDA.

## Correctness tests

Add `tests/test_recurrent_decoder.py` with the following required coverage.

### Configuration and construction

- recurrent mode rejects nonzero terminal decoder depths;
- traversal head dimensions must multiply to `traversal_dim`;
- the model owns eight unique coarse blocks and one shared traversal block;
- parameter count is below the matched terminal model;
- terminal mode remains unchanged during the comparison campaign.

### Tensor and mask contracts

- lengths divisible and nondivisible by four;
- partial tail patches with one, two, and three residues;
- two chains whose segment lengths are not divisible by four;
- residue-index discontinuities;
- prefix padding remains exactly zero after every feedback path;
- atom attention and atom-mean feedback never cross chain segments;
- register tokens are not directly changed by fine-to-coarse feedback.

### Traversal semantics

- unit `k+1` atom input is rebuilt from `atom_skip`, not unit `k` atoms;
- zeroing coarse-to-traversal expansion removes coarse information from the
  scratch residue state without changing `traversal_reference`;
- zeroing single-feedback scales makes coarse singles independent of fine
  activations between units;
- cumulative coordinate updates equal the explicit sum of per-unit scaled
  updates;
- a zero coordinate head gives the exact EDM skip prediction.

### Pair refresh

- geometry refresh matches the eager 16-slot-pair RBF oracle;
- pair masks zero invalid partial-tail combinations;
- transpose-related slot distances are consistent under patch-pair reversal;
- the final compact distogram uses the final refreshed pair state;
- coordinate, geometry projection, and later coarse blocks all receive gradients
  from a final pair loss.

### End-to-end gradients

One loss over final coordinates, sequence logits, and expanded compact
distogram must reach:

- atom input and atom encoder;
- residue encoder and patchify;
- the first and last coarse blocks;
- coarse-to-traversal expansion;
- shared traversal residue attention and FFN;
- shared traversal atom attention and FFN;
- coordinate head and every coordinate scale;
- single-feedback and geometry-feedback scales;
- final sequence and distogram heads.

Because the coordinate head initializes to zero, its input path and coordinate
scales do not all receive nonzero gradients on the exact initial state. Test the
head gradient at initialization, then perturb the head in a separate fixture to
test every coordinate scale and its upstream traversal path. Likewise, test
single-feedback projection gradients after assigning a small nonzero scale,
and separately test that each of the `K-1` scales receives a gradient at zero
initialization.

### Backend parity

On optional CUDA tests, compare eager and fused implementations for:

- geometry-refresh forward and backward;
- factored atom conditioning;
- regular and multichain traversal expansion/reduction;
- BF16 masks and partial patches.

Use tolerances appropriate to BF16 accumulation and require finite gradients.

## Performance validation

### Baseline

The performance comparison uses the current `3/3/8/3/3` broadcast architecture
with the same source feature and self-conditioning cell as the recurrent run.
Both jobs must use:

- one identical H100 allocation where possible;
- batch 32;
- the real filtered length-64/96/128 distribution;
- BF16;
- `torch.compile(dynamic=False)`;
- fused backends required;
- resident owned-shard loading;
- identical losses, optimizer, EMA, and self-conditioning schedule.

### Isolated model profile

Before real-data training, profile forward and forward/backward at exact
`N=64,96,128` shapes. Warm compilation separately, then collect at least 100
iterations per shape. Report:

```text
full_model_ms
encoder_ms
coarse_ms
traversal_residue_ms
traversal_atom_ms
single_feedback_ms
geometry_feedback_ms
loss_ms
backward_ms
optimizer_ms
peak_allocated_mib
top_cuda_events
copy_contiguous_events
```

The profile must distinguish self-conditioned and non-self-conditioned steps.

### Roofline attribution

Capture one warmed `N=128` forward/backward iteration with Nsight Systems, then
use Nsight Compute only on the recurrent kernels that dominate the delta. For
each selected kernel, record:

```text
elapsed_us
launch_count
dram_bytes
dram_throughput_pct
sm_throughput_pct
tensor_core_utilization
achieved_flop_per_s
arithmetic_intensity_flop_per_byte
```

Classify changes from measured counters rather than kernel names:

- high SM/tensor-core utilization with arithmetic intensity above the H100
  ridge point is compute-bound; reduce matrix dimensions or invocation count,
  not memory traffic;
- high DRAM utilization with low SM utilization is bandwidth-bound; remove
  expanded conditions, materialized RBFs, gathers, or redundant writes;
- low SM and DRAM utilization with many short kernels is launch-bound; preserve
  the static compiled loop, fuse adjacent pointwise/reduction work, or evaluate
  CUDA graphs;
- a large `copy_` or `contiguous` delta means the layout contract is broken;
  repair strides or the regular-layout view before writing a custom kernel.

Do not optimize an operator whose inclusive contribution is below 2% of warmed
denoiser time unless it blocks fusion of a larger neighboring region. The
throughput report must include the baseline and recurrent roofline points, not
only the final latency ratio.

### Real-data gate

Run 600 steps with steps 0--99 as warmup. Against the matched terminal baseline:

| metric | target or guardrail |
|---|---:|
| completed-step p50 design target | `<= 1.10x` |
| completed-step p50 rejection guardrail | `<= 1.15x` |
| completed-step p95 rejection guardrail | `<= 1.20x` |
| peak allocated GPU memory guardrail | `<= 1.15x` |
| warmed denoiser forward p50 guardrail | `<= 1.15x` |
| nonfinite losses or gradients | zero |
| eager or unfused fallback | zero |
| unexpected compile after warmup | zero through p95 window |

The target is a throughput guardrail, not permission to hide cost in data
waiting. Report synchronized model, backward, optimizer, and batch-wait phases.

### Sampling gate

The 200-step sampler calls the denoiser repeatedly, so forward latency matters
independently of training. At batch 32 and lengths 64, 96, and 128:

- warmed per-denoiser p50 targets 1.10x and must be at most 1.15x baseline;
- peak sampling memory must fit the existing single-H100 envelope;
- all 200 steps must retain one coherent recurrent external self-conditioning
  call per Euler step;
- sequence and coordinates must be finite for every sampled structure.

## Scientific evaluation

Passing throughput does not promote the architecture. Train recurrent and
terminal models from initialization under a matched contract. Do not initialize
the recurrent model from a terminal checkpoint.

### Short screen

Run at least one matched 25k recurrent/terminal pair using the currently
promoted feature, objective, learning-rate, and self-conditioning configuration.
At 25k, compare:

- coordinate and sequence losses by sigma bucket;
- smooth-lDDT and compact distogram losses;
- gradient norms and clipping fraction;
- per-unit coordinate-update RMS;
- per-unit single- and geometry-feedback RMS;
- pair-state norm by unit;
- fixed-sampler effective alphabet, entropy, maximum residue fraction, and run
  length;
- CA-step, peptide, clash, and stereochemical diagnostics.

Per-unit telemetry must be detached and reduced to scalars. Do not retain full
intermediate coordinates for logging.

### Promotion run

Advance to the matched 100k panel only if:

1. throughput and memory gates pass;
2. losses and gradients remain finite;
3. coordinate-update RMS does not monotonically explode across units;
4. pair norms remain bounded;
5. the 25k fixed-sampler panel shows no material geometry or sequence collapse.

Promotion at 100k requires a better geometry/diversity tradeoff than the matched
terminal model, not merely equal training loss.

## Telemetry

Add optional recurrent diagnostics controlled by the existing logging cadence:

```text
recurrent/unit_00/coordinate_update_rms
recurrent/unit_00/single_feedback_rms
recurrent/unit_00/geometry_feedback_rms
recurrent/unit_00/pair_rms
...
recurrent/unit_07/coordinate_update_rms
```

Units zero through six also log `single_feedback_rms`; unit seven does not
compute that update. All eight units log geometry feedback when a distogram is
requested, while the no-distogram self-conditioning pass omits unit seven.
Also log aggregate minimum, maximum, and final applicable values. Compute
diagnostics only on logging steps and detach before reduction. Production steps
between logs must not materialize diagnostic tensors.

## Implementation sequence

### Phase 1: CPU semantics

1. Add configuration and validation.
2. Implement traversal expansion, shared residue block, atom injection, atom
   mean feedback, cumulative coordinate updates, and eager geometry refresh.
3. Integrate recurrent mode into `HierarchicalKaveh` without changing terminal
   mode.
4. Add CPU tensor, mask, initialization, output, and gradient tests.
5. Confirm no checkpoint migration path exists.

Exit condition: focused CPU tests pass and the small recurrent model has the
same public output shapes as the terminal model.

### Phase 2: CUDA fast path

1. Implement factored atom conditioning while reusing fused atom attention.
2. Extend the fused patch-distance projection to recurrent short-128 shapes.
3. Compile the static eight-unit model with fused backends required.
4. Add CUDA forward/backward parity tests.
5. Inspect traces for condition expansion, `cat`, `contiguous`, gather/scatter,
   and geometry materialization.

Exit condition: no required fast path falls back and isolated parity passes.

### Phase 3: Throughput gate

1. Run paired isolated profiles.
2. Attribute any regression by recurrent substage.
3. Optimize only a measured bottleneck.
4. Run the 600-step real-data gate.
5. Record negative optimizations as well as the accepted path.

Exit condition: all latency, memory, finite-gradient, compile, and fused-backend
limits pass.

### Phase 4: Scientific screen

1. Materialize immutable matched terminal/recurrent configs.
2. Run two-step preflights and verify checkpoints.
3. Train to 25k and run the fixed sampling/analysis panel.
4. Continue to 100k only after the explicit promotion decision.

## Risks and predefined responses

| risk | signal | response |
|---|---|---|
| traversal width is too narrow | underfit sequence/geometry with stable optimization | test 384 only after 256 completes; do not widen silently |
| tied traversal weights limit unit specialization | unit updates become nearly identical | test two alternating shared blocks before eight unique blocks |
| geometry refresh destabilizes pair state | pair RMS or gradient grows by unit | lower the configured initial geometry scale; retain differentiability |
| final distogram becomes a geometry shortcut | distogram improves without topology/sample improvement | ablate final-unit refresh explicitly |
| atom recurrence dominates latency | atom stage exceeds budget in trace | factor conditioning first, then test anchor-only alternating units |
| recurrent graph exceeds memory | peak allocation fails while FLOPs pass | inspect saved tensors, then selectively checkpoint atom FFN |
| compile fragmentation | graph breaks or warmup recompiles recur | register custom ops correctly and make unit count/static shapes explicit |
| shared parameters harm DDP overlap | backward tail or reducer error | test static-graph DDP; use two shared blocks before unique blocks |
| single feedback stays closed | scale remains near zero and coarse state ignores fine path | report scale/update RMS; test small nonzero initialization as an ablation |

## Follow-up ablations, ordered

These are out of scope until the core eight-unit implementation has a complete
profile and 25k result.

1. **Alternating anchor-only atom units.** Run full Atom14 blocks on units 1, 3,
   5, and 7, with a residue-to-C-alpha/CB head on the intervening units.
2. **Two shared traversal blocks.** Alternate block A/B to add unit capacity with
   limited optimizer and communication cost.
3. **Traversal width 384.** Use only if 256 is scientifically under-capacity and
   the measured throughput headroom can absorb it.
4. **Three macrocycles.** Group coarse depths `[3,3,2]` and use only three fine
   excursions if eight-unit recurrence is bandwidth-bound.
5. **Detached geometry feedback.** Consider only as a controlled scientific
   ablation if differentiable recycling causes an activation-memory failure.
6. **FP8 traversal linears.** Consider after BF16 correctness and promotion; keep
   normalization, softmax, geometry, pair state, and coordinate heads in BF16 or
   FP32 as appropriate.

## Promotion and cleanup

If recurrent decoding passes the 100k scientific gate:

1. make it the one supported architecture;
2. remove terminal decoder mode and its unused configuration fields;
3. update `README.md`, `docs/architecture.md`, and the Pallatom conformance
   matrix;
4. regenerate parameter-count and throughput baselines;
5. keep the terminal checkpoints as external experiment artifacts, not as a
   compatibility path in model code.

If it fails throughput, stability, or the 25k scientific screen, remove the
recurrent execution path and retain this document plus the measured result as a
rejected architecture record. Do not leave a default-off unprofiled decoder in
the production model.
