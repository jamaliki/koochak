# Pallatom training conformance

This repository treats Pallatom as the normative diffusion specification, but
the public Pallatom repository contains inference rather than a complete
trainer. Every training choice is therefore classified by evidence instead of
silently inferred.

## Audited primary sources

| source | audited revision | role |
|---|---|---|
| [Pallatom](https://github.com/levinthal/Pallatom) | `b27d70054dec6ce2f5ceadf8977de3d3baf00663` | normative public model and sampler |
| [Pallatom paper](https://proceedings.mlr.press/v267/qu25c.html) | ICML 2025 proceedings | normative training algorithms and hyperparameters |
| [Emyx paper](https://arxiv.org/abs/2606.19377) | arXiv `2606.19377v1` (12 June 2026) | paper-only all-atom architecture and training comparison |
| [ByteDance Protenix](https://github.com/bytedance/Protenix) | `4c355be4553512f72453ecbfb65e69f4c35d1413` | compatible PyTorch EDM trainer cross-check |
| [Protpardelle](https://github.com/ProteinDesignLab/protpardelle) | `dc7c9e42dfbe1a48bcc439485fd4e80f4496b669` | independent all-atom EDM cross-check |
| [Protpardelle-1c](https://github.com/ProteinDesignLab/protpardelle-1c) | `ee378400f25b801fa481028000f9060183d7fb4c` | trainable all-atom EDM and missing-atom cross-check |
| [ByteDance APM](https://github.com/bytedance/apm) | `a98e59b17df29aec9424b598c5fac4bdc492b3e7` | trainable all-atom design comparison; flow matching, not EDM |
| [ByteDance AnewOmni](https://github.com/bytedance/AnewOmni) | `926e99818ea18cf9d9b2064ce0319fe691b7a1f1` | trainable latent biomolecular design comparison; not coordinate EDM |
| [ByteDance DPLM](https://github.com/bytedance/dplm) | `8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d` | trainable multimodal discrete diffusion comparison |
| [ByteDance PXDesign](https://github.com/bytedance/PXDesign) | `f788441313c84c3074fe9596ac2433f96b15c763` | released diffusion-generator runtime; no comparable trainer |
| [Genie 3](https://github.com/aqlaboratory/genie3) | `d77ae5ac04212ff1e8b29b585859a3244c614804` | all-atom architecture/sampler comparison; no comparable public trainer |

APM, AnewOmni, and DPLM are useful design references, but their flow-matching,
latent, or discrete corruption laws are not mixed into this coordinate EDM.

Emyx is especially close at the architectural boundaries: it uses Rep14,
128-dimensional atom states with four attention heads, residue-index-local atom
connectivity, a 768-dimensional token trunk, and gated atom-to-token and
token-to-atom residual transfer. Those independent choices support this
repository's Atom14 encoder/decoder and persistent atom skip. Emyx instead
trains a velocity field on a linear conditional-flow interpolant with
`sigma_data=10`, a Beta/uniform time mixture, time-weighted lDDT, AdamW at
`2e-4`, warmup, and weight decay. None of those incompatible training rules is
imported into the Pallatom EDM contract. No public Emyx training repository was
accessible at the paper's release revision on 13 August 2026, so these claims
are explicitly paper-only.

## Exact matches

| mechanism | implementation |
|---|---|
| training noise | `sigma = sigma_data * exp(N(-1.2, 1.5^2))`, `sigma_data=16` |
| corruption | one isotropic Gaussian noise level per structure; no ambient floor |
| augmentation | center valid atoms, random rigid rotation, translation std 1 A |
| preconditioning | standard EDM `c_in`, `c_skip`, `c_out`; `c_noise=log(sigma/sigma_data)/4` |
| coordinate objective | align target to prediction with stopped-gradient FP32 Kabsch, then MSE divided by `3 * valid_atoms` and weighted by `1/c_out^2` |
| sequence objective | final 20-class cross entropy, weight `0.25`, active only for examples with `sigma <= 0.5 A` by default |
| sequence head | layer-normalize and transform final atom features, mean-pool per residue, zero-initialized 20-class projection |
| smooth lDDT | all Atom14 pairs within 15 A, thresholds 0.5/1/2/4 A, exact chunked evaluation, weight `1.0` |
| residue distogram | directional logits summed with their transpose, 64 bins over 2.3125..21.6875 A, all `i,j` pairs including diagonal, weight `0.5` |
| optimizer | Adam, `lr=1e-3`, betas 0.9/0.999, no warmup, no weight decay |
| reference run | batch 32, crop 128, 300,000 steps, 100% self-conditioning |
| sampler | 200 coherent perturbed-time Euler steps, one denoiser call per step, previous-step coordinate self-conditioning, `tmin=0.01`, `tmax=1`, `gamma=0.2`, noise scale `1.003`, step scale `2.25`, and final temperature-0.1 softmax/argmax sequence decoding; churn is gated on the unperturbed discrete `t/T` fraction as in Pallatom Algorithm 1 |

Protenix independently confirms the scaled log-normal training noise, rigid
augmentation, FP32 stopped-gradient target alignment, and EDM scaling. It is a
cross-check, not a source of substituted AF3 objectives.

Protpardelle-1c independently confirms rigid augmentation, translation std 1 A,
masked coordinate losses, and explicit handling of unresolved and dummy atoms.
Its all-atom `cc91` recipe uses Atom37, `sigma_data` near 10, `P_mean=-0.5`, and
AdamW at `1e-4`; those model-specific choices do not override Pallatom. Its mask
discipline does constrain ingestion here: virtual side-chain slots are distinct
from genuinely unresolved backbone atoms, and a residue without C-alpha is not
silently converted into a valid token.

Emyx independently supports terminal all-atom local-distance supervision and
the Atom14-to-residue-to-Atom14 information path. Its lDDT differs in two
material ways: it excludes same-token pairs and applies a flow-time-dependent
weight. Pallatom remains normative here, so neither Emyx change is copied.

## Explicit interpretations and deviations

- The public Pallatom materials do not specify a sequence-loss noise cutoff.
  Hierarchical Kaveh makes this training factor explicit as
  `loss.aatype_sigma_max`; the default factorial-screen gate is inclusive at
  `0.5 A`, and the active-example fraction is logged.

- The released sampler independently perturbs the time used to initialize the
  coordinates and each subsequent denoiser time. Hierarchical Kaveh instead
  samples one perturbed high-noise start and derives a monotone grid ending at
  zero. Each Euler endpoint is therefore the noise level declared at the next
  denoiser call; this corrects a sampler-state/conditioning mismatch.
- The paper assigns 2x weight to “polar residues” but neither the paper nor
  public code defines the set. The default is explicitly configured as
  `RNDCEQHKSTY`; it can be changed without editing loss code.
- Pallatom self-conditions through a coordinate-derived template distogram.
  Hierarchical Kaveh deliberately has no geometry refresh or node-to-pair path,
  so its 100% self-conditioning supplies the detached first-pass coordinates to
  the persistent atom stream. Pair state remains initialized once.
- The paper's Algorithm 1 can be read as two model evaluations per step, while
  the released sampler performs one evaluation and carries its prediction into
  the next step. The executable Pallatom implementation is authoritative here,
  so this repository follows the released one-call recurrent path.
- Pallatom supervises every decoder unit. Hierarchical Kaveh intentionally emits
  final predictions only, so intermediate loss is absent.
- Pallatom projects a persistent local atom-pair representation to a 22-bin
  atomic distogram. This architecture has structured atom attention but no
  persistent atom-pair tensor, so that auxiliary head is absent. The all-atom
  smooth-lDDT loss remains exact; it is not presented as an atomic-distogram
  replacement.

These deviations are architecture contracts, not hidden fallbacks. Any future
change should update this matrix and add a source-backed test.
