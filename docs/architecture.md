# Architecture contract

The architecture is a five-stage hierarchy with a fixed patch size of four.
Widths and the number of blocks in each stage are scaling knobs; the information
flow is not.

The atom/residue split is independently consistent with Emyx
([arXiv:2606.19377](https://arxiv.org/abs/2606.19377)): both use fixed Atom14
tokens, 128-dimensional four-head local atom processing, a 768-dimensional
residue/token stream, and residual information transfer back into an atom
decoder. Hierarchical Kaveh retains its own p=4 stable pair stream and global
residue stages; it does not adopt Emyx's coordinate-dependent sparse graph,
recycling, or flow-matching objective.

## 1. Atom encoder

Each valid Atom14 slot receives its noisy coordinate, slot identity, residue
features, diffusion-time conditioning, and optional self-conditioned coordinate.
Atom attention is local by residue index: an atom may attend to all Atom14 slots
in its own residue and neighboring residues within `atom_window_radius`, but not
across a chain segment. The encoded atom tensor is saved for the decoder.

Learned masked pooling produces one residue vector. An explicit mean-pooling
residual makes atom-to-residue information flow available from initialization.

## 2. Global residue encoder

Residues and learned registers use global RoPE attention. Prefix-padded batches
are packed for FA3 varlen attention; PyTorch SDPA is the reference fallback.
This is the first global communication stage and occurs before lossy patching.

## 3. Coarse p=4 trunk

Patchify groups four contiguous residues without crossing a chain boundary,
explicit break, residue-index discontinuity, or padding. Learned masked pooling
again sits on an exact mean residual. A tail of one to three residues forms a
masked partial patch rather than being dropped.

The stable pair initializer keeps the 4x4 ordered C-alpha distance matrix for
every patch pair. Sixteen RBF distance channels per slot pair preserve more
geometry than centroid distance. Signed-log residue separation, chain identity,
and slot validity supply static topology.

Every coarse block applies:

1. pair-biased patch attention;
2. node feed-forward update;
3. outgoing pair multiplication;
4. incoming pair multiplication.

There is no triangle attention, pair transition, node-to-pair update, predicted
coordinate refresh, or pair-state recycling. The pair matrix is stable and
compact at `ceil(N/4)^2` scale.

## 4. Residue decoder

Unpatchify broadcasts a slot-aware coarse update onto the saved pre-patch residue
stream. Global residue attention then recovers residue-granular communication.
The skip prevents the p=4 bottleneck from becoming the only route for local
information.

## 5. Atom decoder and outputs

The decoded residue stream is injected into the saved atom encoder tensor. A
second local atom-attention stage resolves Atom14 coordinates while retaining a
direct atom encoder-to-decoder residual.

The sequence head follows Pallatom after the final atom decoder: transformed
atom features are mean-pooled per residue, then projected to canonical residue
logits. Sequence prediction therefore consumes the final local atom reasoning
rather than branching from the pre-atom residue stream.

Only final predictions are emitted:

- denoised coordinates `[B,N,14,3]`;
- amino-acid logits `[B,N,20]` over the canonical residue types;
- a compact patch distogram for training loss and optional dense materialization.

Unknown/mask and padding remain distinct only in the 22-symbol residue *input*
vocabulary; they are not output classes.

## Scaling experiments

Vary `atom_encoder_depth`, `residue_encoder_depth`, `coarse_depth`,
`residue_decoder_depth`, and `atom_decoder_depth` in `ModelConfig`. Keep p=4,
triangle multiplication in every coarse block, the stable pair initialization,
and both skip paths fixed when comparing scaling behavior. Throughput comparisons
must use the same sequence-length distribution, precision, batch occupancy,
gradient mode, optimizer step, and hardware.
