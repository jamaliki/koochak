# Atom4 plus residue-latent campaign ledger

**Status:** completed record of the measured stages available on 2026-08-28.
**Scope:** side-chain codec development, the 50k latent factorial, promoted
and selected16 continuations through 200k, the strict filtered top-four
continuation, ESMFold/all-atom evaluation, and the historical corrected
designability inventory.

This entry is deliberately explicit about evidence maturity. A number is a
point estimate from the stated panel; it is not a claim that generated
proteins are independent biological replicates.

## 1. Executive record

The campaign established four main facts.

1. The fixed side-chain codec is not the bottleneck at reconstruction time:
   the promoted z4 codec has held-out sequence accuracy `0.9999965`, median
   chi error `3.858 deg`, and effective latent rank `3.722`.
2. The Atom4 plus latent denoiser produces meaningful strict designability,
   but length dominates the current endpoint. At 200k across the completed
   selected16 arms, rates were `57.69%` for length 64, `27.16%` for length 96,
   and `15.14%` for length 128.
3. Training maturity changes the ranking rather than producing a single stable
   winner. On the common 12-arm panel, `r2_z2_b0p03_c0` fell from first at 150k
   to ninth at 200k, while `r1_z3_b0p01_c0p125` rose from tenth to first.
4. The historical corrected rescore shows that sampling and sequence/structure
   co-design are major contributors to the gap. Mature Kaveh reached `35/96`
   at its best tested scale; ProteinMPNN rescued `110/128` redesigns on the
   same type of backbone panel; and the recovered old structural sampler
   changed one matched checkpoint from `2/48` to `26/48`.

The appropriate next decision is therefore not “make the codec larger.” It is
to use the best current sampler/clock settings, maintain the length-stratified
endpoint, and run paired sampler and sequence-rescue tests before another broad
architecture sweep.

## 2. Experimental questions and fixed contracts

### 2.1 Codec question

How do latent dimensionality, KL regularization, and chemical hierarchy
regularization affect a fixed-codec representation that will be used by the
downstream denoiser?

Fixed codec data and split:

- candidate-indexed AFDB v6 CIF structure subset;
- candidate table SHA-256
  `7ca475e5dc834d307c0be408f26a0064ef7d573caf8df35e97e115ac8284fce6`;
- materialized residue NPZ SHA-256
  `e804fa391cceed404e4c14a23c8881f4c64c3069b3828aedd5d8a0b6dbb4c066`;
- 28,100 structure candidates: 22,487 train, 2,829 validation, 2,784 test;
- residue counts: 2,318,932 train, 291,349 validation, 288,583 test;
- per-amino-acid template balancing at 2,000 residues per identity;
- split by `sha256(seed:model_id) mod 10`, with residues 0-7 train, 8
  validation, and 9 test, seed 42.

### 2.2 Denoiser question

Can a fixed side-chain latent be predicted jointly with a four-atom backbone
using the existing hierarchical Kaveh architecture, and which codec settings
remain competitive after end-to-end diffusion training?

Fixed latent-factorial contract:

- state: `N, CA, C, O` plus one residue latent;
- atom-local latent conditioning and atom-local latent output, averaged per
  residue;
- target: frozen codec posterior mean;
- latent transform: training-mean centered and covariance whitened;
- training latent clock: `kappa=1` for half of examples and otherwise
  log-uniform delay in `[1, 4]` relative to backbone noise;
- inference schedule: `constant_k3`;
- data view: resolved lengths 4-128, mean pLDDT `>80`, loop content `<0.5`;
- baseline factorial does **not** apply packing-density or maximum-loop-length
  filters;
- one H100, batch 256, no DDP, no gradient accumulation, 50,000 steps;
- evaluation: 32 samples each at lengths 64, 96, and 128 per arm.

## 3. Exact codec formulation

### 3.1 SidechainVAE input/output

For each residue, the encoder input is

```text
x = [one_hot(aatype, 20), chi_vector_components(4 x 2), chi_mask(4)]
```

so the input width is `20 + 8 + 4 = 32`. The network is:

```text
encoder:  Linear(32, 64) -> SiLU -> Linear(64, 2d)
          split into posterior mean mu[d] and log variance logvar[d]
          clamp logvar to [-12, 8]

sample:   z = mu + exp(0.5 * logvar) * epsilon
          epsilon ~ N(0, I); evaluation/no-grad uses z = mu

decoder:  Linear(d, 64) -> SiLU -> Linear(64, 28)
          first 20 outputs are amino-acid logits
          final 8 outputs are four raw (cos chi, sin chi) vectors
          each chi vector is normalized and masked by residue identity
```

The parameter count is exactly

```text
(32*64 + 64) + (64*2d + 2d) + (d*64 + 64) + (64*28 + 28)
= 3996 + 194d
```

| Latent d | Parameters |
| ---: | ---: |
| 2 | 4,384 |
| 3 | 4,578 |
| 4 | 4,772 |
| 5 | 4,966 |
| 8 | 5,548 |
| 12 | 6,324 |
| 16 | 7,100 |

### 3.2 Base VAE loss

Let `m_chi` select resolved torsions and let `u_hat` and `u` be normalized
predicted and target chi vectors. The implemented losses are:

```text
L_aa   = weighted_cross_entropy(identity_logits, aatype)
L_chi  = mean_{resolved chi} [1 - dot(u_hat, u)]
L_norm = mean_{resolved chi} (||raw_chi_hat|| - 1)^2
L_KL   = mean_residue [ -0.5 * sum_j(1 + logvar_j - mu_j^2 - exp(logvar_j)) ]

L_base = L_aa + 2.0 L_chi + 0.01 L_norm + beta L_KL
```

Class weights are `sqrt(mean(counts) / clamp(counts, 1))`, normalized to mean
one. This is a reconstruction-plus-regularization objective, not a direct
structural designability objective.

### 3.3 Chemical hierarchy loss

The residue order is `ARNDCQEGHILKMFPSTWYV`, and the source matrix is BLOSUM62.
For residues present in a batch, posterior-mean centroids are

```text
c_a = mean(mu_i | aatype_i = a)
```

The BLOSUM-derived target distance is the Euclidean distance induced by the
similarity kernel:

```text
D_B(a,b) = sqrt(max(0, B_aa + B_bb - 2 B_ab))
```

For latent centroid distance `D_z`, the implementation uses squared latent
distances for the neighborhood logits. With off-diagonal entries only,

```text
s_z2 = detach(mean(D_z^2))
q_B(a, .) = softmax(B(a, .) / T_B)
q_z(a, .) = softmax(-D_z(a, .)^2 / (s_z2 T_z))
L_neighbor = mean_a KL(q_B(a, .) || q_z(a, .))
```

The stress term compares normalized distances using smooth L1:

```text
s_z = detach(mean(D_z))
s_B = mean(D_B)
L_stress = smooth_l1(D_z / s_z, D_B / s_B)
```

The requested chemical value is a **gradient-RMS ratio**, not a raw coefficient.
At the first continuation batch, scales are calibrated so the chemical gradient
RMS is the requested ratio of the base VAE gradient RMS. With
`chemical_stress_fraction = 0.25`:

```text
chemical gradient budget = 0.75 neighborhood KL + 0.25 stress
scale_neighbor = ratio * 0.75 * base_grad_rms / neighbor_grad_rms
scale_stress   = ratio * 0.25 * base_grad_rms / stress_grad_rms
L_total       = L_base + ramp * (scale_neighbor L_neighbor + scale_stress L_stress)
```

The chemical ramp is linear over the first 25% of each six-epoch continuation;
the continuation has no VAE warmup. Both hierarchy temperatures are 1.0.

### 3.4 Codec grids and selected checkpoints

The base factorial contains `3 replicates x 4 dimensions x 4 beta values x 4
chemical ratios = 192` end-to-end codec variants:

| Factor | Levels |
| --- | --- |
| Replicate seed | 42, 43, 44; continuation seeds 1042, 1043, 1044 |
| Latent dimension | 2, 3, 4, 5 |
| Parent beta | 0.01, 0.03, 0.1, 0.3 |
| Chemical gradient ratio | 0, 0.125, 0.25, 0.5 |

Parent training used 12 epochs, batch 8192, Adam learning rate `3e-3`. Each
parent was continued for six epochs at learning rate `1e-3`; the four chemical
continuations share parent checkpoint, seed, data order, and final-checkpoint
selection. `chi_weight=2.0`, BLOSUM temperature 1.0, latent temperature 1.0.

A separate large screen contains 48 variants: dimensions 4, 8, 12, 16; beta
values `1e-5, 1e-4, 1e-3, 1e-2`; chemical ratios `0, 0.125, 0.25`; three
replicates. The promoted large-screen cells were:

| Cell | d | beta | chemical ratio | Params | KL nats/res | Test seq. acc. | Median chi error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `large_z4_b0p01_c0p25` | 4 | .01 | .25 | 4,772 | 9.1801 | .9999792 | 2.6786 deg |
| `large_z8_b0p01_c0p125` | 8 | .01 | .125 | 5,548 | 9.9355 | .9999896 | 2.2410 deg |
| `large_z8_b0p01_c0p25` | 8 | .01 | .25 | 5,548 | 11.7471 | .9999931 | 2.3570 deg |
| `large_z12_b0p01_c0p25` | 12 | .01 | .25 | 6,324 | 14.1813 | 1.0000 | 2.3555 deg |

The canonical promoted z4 codec manifest records:

- checkpoint SHA-256 `837912722dd421e6029043c68159eeb1f4278ec24dc7bcf2dc7f9f6afe0c829c`;
- `d=4`, hidden width 64, 4,772 parameters, beta `.03`, chemical ratio `.175`;
- posterior-mean variances
  `[0.7960190145, 1.3810674244, 1.3918115254, 1.6528865003]`;
- latent data scale `sqrt(mean variance) = 1.1425612089230712`;
- held-out sequence accuracy `.9999965429`;
- held-out median chi error `3.858463 deg`;
- held-out BLOSUM-distance Spearman `.6259601`;
- held-out effective rank `3.7218002`.

Campaign launch manifests used immutable replicate-specific checkpoints. For
example, the successful baseline factorial z4 preflight used codec SHA-256
`337aced79bcf449280ca28a36f1143b18a15299bedb001e3ad8a72d74a8595c1`, while
the filtered-top4 z4 preflight used
`4a1ae443543bcb1359bd6ee2707a26a8186de7f83f3ccd62b3bd240b97c176bb`. These
are not interchangeable with the later canonical manifest; the SHA in each
launch manifest is the authority for that run.

For the factorial and filtered denoisers, the latent diffusion space was
standardized with `latent_data_scale=1.0` and the codec manifest’s
replicate-specific posterior-mean center and whitening matrix. The filtered
z4 preflight recorded center
`[0.05768133, -0.09571360, -0.07090995, 0.09748257]` and whitening matrix

```text
[[ 0.71379578, -0.01781956, -0.12494326,  0.03684827],
 [-0.01781956,  0.73949552, -0.04321425, -0.00614978],
 [-0.12494326, -0.04321425,  0.96520871, -0.02725539],
 [ 0.03684827, -0.00614978, -0.02725539,  0.56435722]]
```

The per-arm center, whitening matrix, codec checkpoint, and checksum remain in
the immutable codec manifests; the table above is the exact filtered z4
preflight contract, not a replacement for those per-arm records.

## 4. Production model architecture

The model is a five-stage hierarchy with fixed patch size four and a stable
pair stream. The production selected16/filtered configuration is:

| Setting | Value |
| --- | ---: |
| Node / condition / pair / atom widths | 768 / 256 / 64 / 128 |
| Residue attention | 12 heads x 64 dimensions |
| Atom attention | 4 heads x 32 dimensions |
| Atom encoder / residue encoder / coarse / residue decoder / atom decoder | 1 / 4 / 6 / 4 / 1 blocks |
| Atom local window radius | 1 residue |
| Residue / atom FFN expansion | 4 / 2 |
| Dropout | 0.2 |
| Pair RBFs and distance range | 16, 0.05-22 A |
| Distogram | 64 bins, 2.3125-21.6875 A |
| Learned registers | 4 |
| Coordinate sigma data | 16.0 A |
| Latent sigma data | run-specific; canonical z4 1.1425612 |
| Atom latent conditioning/output | enabled / enabled |

Information flow:

1. Each `N,CA,C,O` slot receives noisy coordinates, slot identity, residue
   features, coordinate/latent time conditions, and self-conditioned
   coordinates. Atom attention is local within the residue and neighboring
   residues, without crossing chain breaks. One masked mean residual pools atom
   states into residues.
2. Residues and four learned registers use global RoPE attention before
   patching. This is the first global communication stage.
3. Patchify groups four residues without crossing chain boundaries, residue
   index discontinuities, or padding. Tail residues form a masked partial patch.
   The pair initializer keeps the ordered 4x4 C-alpha distance matrix for each
   patch pair, with 16 RBF channels per slot pair plus signed-log sequence
   separation, chain identity, and slot validity.
4. Every one of six coarse blocks applies pair-biased patch attention, node FFN,
   outgoing pair multiplication, and incoming pair multiplication. There is no
   triangle attention, pair transition, node-to-pair update, coordinate refresh,
   or pair-state recycling.
5. Unpatchify broadcasts a slot-aware coarse update to the saved residue stream;
   four global residue decoder blocks restore residue-granular communication.
6. The saved atom encoder state is combined with decoded residues, then one
   local atom decoder resolves Atom4 coordinates. The atom-local latent output
   averages back to one latent per residue.
7. The frozen codec decoder maps predicted latent to residue logits and four
   normalized chi vectors. The denoiser emits Atom4 coordinates, latent, fixed
   decoder outputs, and a compact patch distogram.

Latent coupling is smoothstep-like: zero above backbone sigma 5 A and full by
sigma 2 A. The latent target is frozen posterior mean; codec parameters are
excluded from optimizer and EMA. The joint loss is stopped-gradient Kabsch-
aligned Atom4 EDM MSE plus latent EDM MSE, smooth Atom4 lDDT, and CA distogram
loss weighted 0.5. Identity/chi decoder outputs were logged without gradient in
the first production run rather than used as a decoded Cartesian lDDT target.

The filtered preflight measured 150,547,025 trainable model parameters and
4,772 frozen codec parameters. It used the same core architecture but a
run-specific latent transform and latent scale recorded in its launch report.

## 5. Designability endpoint and analysis pipeline

The evaluator sequence is:

1. sample Atom4 plus latent;
2. decode latent to amino-acid identity and torsions with the frozen codec;
3. write the generated backbone PDB and sequence;
4. run ESMFold to obtain an Atom37 PDB and per-atom pLDDT;
5. align ESMFold to generated coordinates using CA atoms only;
6. compute strict designability as CA RMSD `<2 A` plus actual-atom mean pLDDT
   `>80`;
7. separately compute matched physical Atom14 heavy-atom RMSD with terminal
   symmetry corrections for ASP, GLU, PHE, and TYR.

The selected panels have 96 samples per arm: 32 each at lengths 64, 96, and
128. These are adequate for ranking signals but have substantial binomial
uncertainty. The all-atom metric is a secondary structural consistency metric;
it does not replace the primary endpoint.

## 6. Latent-factorial and continuation results

### 6.1 Original 50k factorial maturity

The 50k factorial launch requested 192 codec/diffusion variants. The available
remote state contains 78 sampled/analyzed arms and 7,488 preliminary ESMFold
predictions, but the intended pooled `factorial-analysis/summary.json` was not
finalized: that directory contains only its launch manifest. Therefore no
pooled 50k factorial designability rate is reported here.

The 50k preliminary rows that were available for selected/promoted arms include:

| Arm | Designable | Rate |
| --- | ---: | ---: |
| `r0_z5_b0p03_c0` | 12/96 | 12.50% |
| `r1_z2_b0p3_c0p25` | 12/96 | 12.50% |
| `r1_z3_b0p01_c0p125` | 15/96 | 15.63% |
| `r1_z4_b0p03_c0p25` | 15/96 | 15.63% |
| `r1_z4_b0p3_c0p5` | 18/96 | 18.75% |
| `r2_z2_b0p1_c0` | 6/96 | 6.25% |
| `r2_z4_b0p03_c0p25` | 12/96 | 12.50% |
| `r0_large_z4_b0p01_c0p25` | 9/96 | 9.38% |
| `r1_large_z4_b0p01_c0p25` | 9/96 | 9.38% |
| `r1_large_z8_b0p01_c0p125` | 2/96 | 2.08% |
| `r1_large_z8_b0p01_c0p25` | 2/96 | 2.08% |

This is a partial table, not a complete 16-arm selected16 aggregate.

### 6.2 Selected16 at 150k and 200k

The 150k recovery summary contains 12 arms, 1,152 samples, and 354 strict
designable samples (`30.7292%` overall). Three selected16 arms had no completed
200k sample set and are not silently counted as failures.

| 150k rank | Arm | Designable | Mean pLDDT | Median CA RMSD | Median all-atom RMSD |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `r2_z2_b0p03_c0` | 38/96 (39.58%) | 76.90 | 1.2470 A | 1.6514 A |
| 2 | `r1_z4_b0p03_c0p25` | 33/96 (34.38%) | 74.56 | 1.3650 A | 1.7983 A |
| 3 | `r0_z2_b0p1_c0` | 32/96 (33.33%) | 74.34 | 1.4893 A | 1.9393 A |
| 3 | `r1_z2_b0p3_c0p25` | 32/96 (33.33%) | 74.87 | 1.3440 A | 1.6930 A |
| 3 | `r2_z2_b0p1_c0p125` | 32/96 (33.33%) | 72.57 | 1.3839 A | 1.8131 A |
| 6 | `r0_large_z4_b0p01_c0p25` | 31/96 (32.29%) | 76.77 | 1.2755 A | 1.5936 A |
| 7 | `r0_z3_b0p3_c0p125` | 30/96 (31.25%) | 73.80 | 1.3172 A | 1.6856 A |
| 8 | `r0_z5_b0p03_c0` | 29/96 (30.21%) | 74.95 | 1.3848 A | 1.7431 A |
| 9 | `r1_z3_b0p01_c0p125` | 29/96 (30.21%) | 75.38 | 1.5462 A | 1.8226 A |
| 10 | `r2_z4_b0p03_c0p25` | 29/96 (30.21%) | 75.38 | 1.3465 A | 1.7294 A |
| 11 | `r1_z4_b0p3_c0p5` | 23/96 (23.96%) | 71.07 | 1.5021 A | 1.9216 A |
| 12 | `r1_large_z8_b0p01_c0p25` | 16/96 (16.67%) | 67.76 | 1.8122 A | 2.2232 A |

The completed 200k recovery summary contains 13 arms, 1,248 samples, and 416
strict designable samples (`33.3333%` overall). It includes all arms for which
200k samples existed; three other selected16 training arms had no 200k samples:
`r0_large_z8_b0p01_c0p125`, `r1_large_z4_b0p01_c0p25`, and
`r1_large_z8_b0p01_c0p125`.

| 200k rank | Arm | Designable | Mean pLDDT | Median CA RMSD | p90 CA RMSD | Median all-atom RMSD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `r1_z3_b0p01_c0p125` | 44/96 (45.83%) | 77.06 | 1.1162 A | 2.6676 A | 1.4542 A |
| 2 | `r1_z4_b0p03_c0p25` | 43/96 (44.79%) | 78.14 | 1.0930 A | 2.5869 A | 1.4032 A |
| 3 | `r0_z3_b0p3_c0p125` | 41/96 (42.71%) | 76.32 | 1.2332 A | 3.8662 A | 1.6481 A |
| 4 | `r0_z2_b0p1_c0` | 37/96 (38.54%) | 75.43 | 1.3162 A | 2.6503 A | 1.7384 A |
| 5 | `r2_z4_b0p03_c0p25` | 36/96 (37.50%) | 76.04 | 1.2080 A | 4.6527 A | 1.6391 A |
| 6 | `r1_z4_b0p3_c0p5` | 34/96 (35.42%) | 76.00 | 1.2306 A | 3.1428 A | 1.6003 A |
| 7 | `r1_z2_b0p3_c0p25` | 32/96 (33.33%) | 73.15 | 1.2532 A | 9.9290 A | 1.6563 A |
| 8 | `r0_large_z4_b0p01_c0p25` | 31/96 (32.29%) | 76.15 | 1.3855 A | 2.6801 A | 1.6532 A |
| 9 | `r2_z2_b0p03_c0` | 29/96 (30.21%) | 74.04 | 1.3212 A | 3.4105 A | 1.7272 A |
| 10 | `r0_z3_b0p03_c0p5` | 27/96 (28.13%) | 73.68 | 1.3616 A | 6.4288 A | 1.7594 A |
| 11 | `r2_z2_b0p1_c0p125` | 24/96 (25.00%) | 71.03 | 1.4239 A | 11.0345 A | 1.8365 A |
| 12 | `r0_z5_b0p03_c0` | 22/96 (22.92%) | 74.21 | 1.3965 A | 2.6758 A | 1.7142 A |
| 13 | `r1_large_z8_b0p01_c0p25` | 16/96 (16.67%) | 69.25 | 1.7076 A | 4.7785 A | 2.1478 A |

Across the common 12 arms, the 150k to 200k designability changes in
percentage points were:

| Arm | Delta |
| --- | ---: |
| `r1_z3_b0p01_c0p125` | +15.63 |
| `r1_z4_b0p3_c0p5` | +11.46 |
| `r0_z3_b0p3_c0p125` | +11.46 |
| `r1_z4_b0p03_c0p25` | +10.42 |
| `r2_z4_b0p03_c0p25` | +7.29 |
| `r0_z2_b0p1_c0` | +5.21 |
| `r0_large_z4_b0p01_c0p25` | 0.00 |
| `r1_large_z8_b0p01_c0p25` | 0.00 |
| `r1_z2_b0p3_c0p25` | 0.00 |
| `r0_z5_b0p03_c0` | -7.29 |
| `r2_z2_b0p1_c0p125` | -8.33 |
| `r2_z2_b0p03_c0` | -9.38 |

The movement is too large for a single “best codec” story. It supports
continuing `r1_z3_b0p01_c0p125` and `r1_z4_b0p03_c0p25`, but only with matched
seeds and repeated panels.

### 6.3 Length effect at 200k

Pooled over the 13 completed arms:

| Length | Samples | Designable | Rate | Mean per-arm median CA | Mean per-arm median all-atom |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 416 | 240 | 57.69% | 0.9209 A | 1.3683 A |
| 96 | 416 | 113 | 27.16% | 1.3132 A | 1.6866 A |
| 128 | 416 | 63 | 15.14% | 2.2496 A | 2.6230 A |

The length gradient is the clearest remaining failure mode. It argues for
reporting 64/96/128 separately and not optimizing a pooled metric that can hide
long-protein collapse.

## 7. Strict filtered top-four experiment

The filtered campaign added all three requested strict filters to the baseline
mean-pLDDT and length view:

```text
loop_content < 0.5
maximum_loop_length < 15
packing_density > 0.3
```

The preflight reference count was 33,642, with length buckets 345 (64), 9,931
(96), and 23,366 (128). The selected arms were:

```text
r1_z4_b0p03_c0p25
r1_z2_b0p3_c0p25
r1_large_z4_b0p01_c0p25
r2_z2_b0p03_c0
```

At 50k, all four arms completed the all-atom summary: 384 samples, 148
designable, `38.5417%` overall.

| Rank | Arm | Designable | Mean pLDDT | Median CA RMSD | Median all-atom RMSD |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `r2_z2_b0p03_c0` | 42/96 (43.75%) | 75.58 | 1.2671 A | 1.8128 A |
| 2 | `r1_z2_b0p3_c0p25` | 39/96 (40.63%) | 76.52 | 1.3385 A | 1.7327 A |
| 3 | `r1_z4_b0p03_c0p25` | 35/96 (36.46%) | 73.76 | 1.2973 A | 1.7140 A |
| 4 | `r1_large_z4_b0p01_c0p25` | 32/96 (33.33%) | 72.34 | 1.4466 A | 1.7663 A |

The filtered 100k, 150k, and 200k all-atom summaries were absent at the last
verified state; the continuation was therefore not treated as a completed
filtered maturity result. One 50k sample-analysis task initially failed from
an exact Scruffy GPU-mask mismatch (`job-3e3b68893f61111bc452`) and was
successfully rerun as `job-ac9d36ba3cb9e0a553b4`. The recovered arm’s sample
metrics were entropy `3.146672 bits`, effective alphabet `9.06856`, latent
effective rank `3.86507`, latent norm mean `1.67431`, and CA-step bad fraction
`0.000273691`.

## 8. Failures, recoveries, and maturity controls

- The first selected16 200k ESMFold workflow failed before inference because
  `constant_k3` was passed without `--schedule-family refinement`, producing
  `ValueError: unknown standard schedules: ['constant_k3']`. The recovery
  workflow added the family flag and completed 13 ESMFold arms plus the
  all-atom aggregate.
- Two selected16 200k sample-analysis jobs failed with Scruffy GPU mapping
  mismatch rather than model errors: `r0_z2_b0p1_c0` job
  `job-456f731fff6e55a3825b` and `r1_z2_b0p3_c0p25` job
  `job-56284465bd72784ff8ac`. The recovered summaries were used.
- The 150k arm `r2_z2_b0p03_c0` required an ESMFold retry; the full12 all-atom
  recovery completed with 354/1,152 designable samples.
- The filtered 50k GPU-mask failure above was repaired by a dedicated recovery
  task. This is why its row is included in the 50k aggregate.
- Three selected16 200k training/sample arms never produced a 200k sample set:
  `r0_large_z8_b0p01_c0p125`, `r1_large_z4_b0p01_c0p25`, and
  `r1_large_z8_b0p01_c0p125`. Their absence is missingness, not a zero score.
- The all-atom RMSD implementation was checked against the CA-only reference;
  maximum CA RMSD regression error was `0.000121 A`.

## 9. Historical corrected designability results

The broad historical inventory is retained in the canonical rescore entry and
machine-readable JSON linked from the index. The key corrected results are
reproduced here because they change the interpretation of this campaign.

### 9.1 Corrected evaluator and sampler findings

- Old pLDDT averaged zero-filled Atom37 slots, so old mean pLDDT and strict
  pass/fail labels were invalid. CA RMSD, sequences, clashes, and generated
  structures were unaffected.
- Released Pallatom reached 29/32 (90.6%) at gamma `.2`, scale `2.25`, and
  30/32 (93.8%) at scale `3.25` after correction.
- Mature Kaveh’s exact seven-scale panel rose to 35/96 (36.5%) at scale `2.5`.
- On the matched old-Kaveh `f0000` checkpoint, the current sampler gave 2/48
  (4.2%) while the recovered old structural sampler gave 26/48 (54.2%). The
  paired bootstrap interval for the difference was +35.4 to +64.6 percentage
  points.

### 9.2 Historical inventory of corrected panels

| Campaign | Unique panels | Predictions | Old strict | Corrected strict | Best corrected panel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current sampler step scale | 91 | 8,736 | 82 | 510 | 35/96 |
| Coarse relative-position batch-256 | 12 | 384 | 0 | 6 | 1/32 |
| L128 offset clocks | 108 | 3,456 | 60 | 436 | 15/32 length slice |
| Legacy-quadrature training family | 18 | 288 | 4 | 37 | 6/16 length slice |
| Legacy-quadrature L128 expansion | 90 | 1,440 | 17 | 137 | 6/16 length slice |
| Delayed-sidechain sampler screen | 13 | 480 | 23 | 111 | 46/64, ProteinMPNN |
| EDM causal screen | 18 | 1,728 | 33 | 180 | 59/96, ProteinMPNN |
| EDM two-stage screen | 12 | 1,152 | 78 | 325 | 75/96, ProteinMPNN |
| Fixed quadrature checkpoints | 32 | 1,024 | 6 | 17 | 6/32 |
| Legacy quadrature late milestones | 8 | 256 | 4 | 14 | 6/32 |
| Legacy quadrature maturation | 56 | 1,792 | 1 | 16 | 7/32 |
| Legacy quadrature noise caps | 8 | 256 | 0 | 0 | 0/32 |
| Old-Kaveh training factorial | 189 | 3,024 | 37 | 211 | 10/16 length slice |
| Pallatom initial-noise screen | 1 | 32 | 10 | 28 | 28/32 |
| Pallatom reference lengths | 3 | 96 | 6 | 79 | 28/32 |
| Pallatom release hyperparameters | 5 | 160 | 26 | 128 | 30/32 |
| Pallatom two-pass screen | 1 | 32 | 2 | 28 | 28/32 |
| Pallatom timing screen | 3 | 96 | 4 | 28 | 12/32 |
| Obsolete ProteinMPNN wrapper | 7 | 896 | unavailable | unavailable | no adjacent RMSD rows |
| Sampler recurrence | 5 | 320 | 4 | 30 | 9/64 |
| Early step-scale screen | 7 | 224 | 3 | 15 | 6/32 |
| ProteinMPNN step-scale rescue | 7 | 896 | 181 | 554 | 110/128 per sequence |
| Recycling-stability factorial | 8 | 256 | 0 | 0 | 0/32 |

The filesystem audits found 835 discovered panels and 30,508 predictions;
signature deduplication retained 680 distinct panels, 25,557 predictions, and
125 duplicate-signature groups. These pooled counts are an inventory, not one
binomial experiment.

### 9.3 ProteinMPNN rescue

At mature Kaveh scale 2.25, the original co-designed sequences gave 11/32
direct successes, while four ProteinMPNN sequences per fixed backbone gave
110/128 (85.9%) per sequence and 31/32 (96.9%) with at least one successful
redesign. The latter is an oracle-over-four diagnostic, not an independent
designability rate. No directly successful backbone was lost under best-of-four
redesign. This localizes a substantial part of the remaining gap to
sequence/structure co-design, while leaving a genuine backbone and sampling
component.

## 10. Provenance

### Local source commits

- architecture contract and model code: current campaign source branch;
- codec implementation: `hierarchical_kaveh/sidechain/vae.py` and
  `hierarchical_kaveh/sidechain/chemistry.py`;
- latent model: `hierarchical_kaveh/model/network.py`,
  `hierarchical_kaveh/config.py`, and the Atom4 data/training pipeline;
- 50k factorial launch source: commit `93122c3b774df64395b7c8b28e0867f8889a08f7`;
- promoted large-screen launch source: commit
  `670b98ef7962e9165f0c8962e651fb3d7ad5f4c0`;
- selected16 200k launch source: commit `49b037f10a9916b6f8ad8bfc1da07928ff47411f`;
- filtered top-four launch source: commit
  `c0ce3ea536089dd37fa4e0c2a22f270b65e924e9`.

### Remote result roots

The roots below are provenance pointers on Sandpit Tokyo; result files remain
on the cluster and are not assumed to be available in a fresh clone.

- 50k factorial: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-factorial-50k/93122c3b774df64395b7c8b28e0867f8889a08f7`;
- selected16 150k/200k: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-selected16-200k/49b037f10a9916b6f8ad8bfc1da07928ff47411f`;
- filtered top-four: `/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/atom4-latent-filtered-top4-200k/c0ce3ea536089dd37fa4e0c2a22f270b65e924e9`.

Key summaries:

- selected16 150k: `all-atom-rmsd-recovery/step0150000-full12/summary.json`;
- selected16 200k: `all-atom-rmsd-recovery/step0200000-13arms/summary.json`;
- filtered 50k: `all-atom-rmsd/step0050000/summary.json`;
- filtered recovered sample analysis:
  `sample-analysis-recovery/step0050000/r1_large_z4_b0p01_c0p25/final.json`.

Historical rescore provenance:

- evaluator correction source commit `9c993afa1ac77448d0df3d0d0abbcb0f8b7e2e65`;
- corrected reduced artifact SHA-256
  `90afb891dd26ffdb945982995869d859178c4cec811c8528ad2d985ccd0e60ad`;
- GBI audit artifact hashes: `audit.json`
  `618c2404aa599c028636295c5ae1832fc106816088fa2a456c8522369a5a1ffb`,
  `rows.jsonl`
  `77879cd18e45be7821fdacad5017939fdb691e08efd0fce073abca3ee5c4d320`;
- Lustre audit artifact hashes: `audit.json`
  `36e271af05769d1222cfb4ac6a0cc58991527008b2b0cdc9d4f7c67919273907`,
  `rows.jsonl`
  `1703efeb541a5ab62deaaaede4be06904ec740df84f2cd8b0a00ca98e6c09cf6`.

## 11. Interpretation and next experiments

The strongest evidence-backed interpretation is:

- **Codec:** reconstruction is excellent; z4 captures useful chemical
  organization, and larger z8-z12 codecs improve reconstruction slightly but
  have not yet demonstrated better end-to-end designability.
- **Training:** 200k is beneficial for several arms, but not monotonically for
  every arm. Ranking instability means a single 96-sample panel is not enough
  to promote a codec.
- **Length:** the 128-residue regime is the dominant failure mode and should be
  treated as a separate target.
- **Sampling:** corrected historical results show sampler quadrature, churn,
  scale, and recurrence can move designability by tens of percentage points.
- **Sequence:** ProteinMPNN rescue shows many backbones are sequence-saveable;
  raw co-design remains a separate bottleneck.

Recommended next experiments, in order:

1. Run a paired sampler ablation on the two 200k leaders, changing one sampler
   component at a time around the recovered old structural sampler: integration
   steps, rho, sigma endpoints, scale schedule, and recurrence windows. Keep
   seeds, backbones, ESMFold version, and panel lengths fixed.
2. Repeat the two leading arms, `r1_z3_b0p01_c0p125` and
   `r1_z4_b0p03_c0p25`, with at least three matched sample panels before
   expanding the codec grid. Report Wilson intervals or bootstrap intervals,
   not only point rates.
3. Treat the filtered top-four campaign as incomplete until 100k, 150k, and
   200k summaries exist; then compare filtered versus baseline by matched arm,
   matched length, and matched seed.
4. Add a two-stage evaluation: raw decoded sequence, ProteinMPNN redesign, and
   best-of-four only as a clearly labeled rescue/oracle diagnostic. Do not mix
   rescue rates into raw designability.
5. For the long-protein problem, first use length-specific sampler tuning and
   a 128-residue canary. Only after the paired sampler test should the model
   architecture or latent dimension be expanded.
