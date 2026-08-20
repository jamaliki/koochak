# Delayed-sidechain compact offset clock: 24-arm results

## Scope and decision summary

This entry closes the two-wave compact offset-clock campaign. The first wave
contains the four scaffold-Cbeta ratio/offset cells sampled at 50k, 100k,
150k, and 200k. The second wave adds 20 cells covering the wider clock
surface, delayed Cbeta, clock-gated objectives, onset, and lDDT gate windows;
its complete panel is at 200k.

The main decisions are:

* **Do not replace the legacy `k8_cb_scaffold` 150k checkpoint.** It still has
  the best observed combination of sequence diversity and all-atom and
  side-chain clash proxies. No compact-clock cell dominates it.
* **Use `ratio3_offset0p2` as the compact-clock reference.** Its effective
  alphabet improves from 9.244 at 50k to 9.979 at 200k while all-atom clashes
  fall from 0.339 to 0.274 per residue. At 200k it is the best first-wave
  balance, and it is stable from 100k onward.
* **Treat `ratio2_offset0p1` as a diversity-only alternative.** It reaches an
  effective alphabet of 10.110 at 200k, but its all-atom and side-chain clash
  proxies regress to 0.327 and 0.267 after their 100k-150k minima.
* **Do not promote clock-gated lDDT alone yet.** It produces the best terminal
  sequence panel in the compact screen (effective alphabet 10.854), but its
  side-chain clash proxy rises to 0.307. The sequence gain and geometry loss
  expose an objective tradeoff rather than a schedule-only solution.
* **Reject the combined lDDT and side-chain-self-conditioning gate.** It has a
  0.0324 Calpha-Cbeta failure fraction, effective alphabet 7.578, and 0.416
  side-chain clashes per residue.

These are fixed-panel, single-training-seed comparisons. Clash counts and
bond thresholds are diagnostic proxies, not chemistry-qualified validation or
evidence of foldability.

## Fixed training, clock, and sampling contract

All cells retain the filtered `3/3/8/3/3` local-center model, no Kabsch
alignment, polar weight 2, sequence sigma ramp 0.5 to 1, compiled training,
and the validated length, pLDDT, loop, and packing filters. Cbeta remains on
the backbone clock unless a cell name contains `cbdelayed`.

For backbone sigma between the offset and the fixed 10 A onset, the compact
clock uses

```text
u = clip(log(sigma_bb / offset_sigma) / log(10 / offset_sigma), 0, 1)
b(u) = 64 u^3 (1-u)^3
sigma_sc = sigma_bb * peak_noise_ratio ** b(u)
```

The extra delay is exactly off below the offset and above the onset. Sampling
uses EMA weights, 200 uncompiled Euler steps, EDM rho 5 to sigma 0.003, step
scale 2.25, churn 0.2, BF16, and seed `20260820`. Every variant panel contains
96 samples: 32 each at lengths 64, 96, and 128.

Higher effective alphabet and entropy and lower alanine and maximum-residue
fractions are preferable. Lower clash and bond-bad fractions are preferable.

## First-wave milestone results

Values are means over 96 samples per row.

| Step | Arm | Alanine | Effective alphabet | Entropy | Max fraction | CA clashes/res. | CA-CB bad | All-atom clashes/res. | Side-chain clashes/res. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | `ratio2_offset0p1` | 0.321 | 8.435 | 3.055 | 0.327 | 0.000326 | 0.000085 | 0.322 | 0.236 |
| 50k | `ratio2_offset0p2` | 0.255 | 8.928 | 3.135 | 0.266 | 0.000407 | 0.000367 | 0.438 | 0.349 |
| 50k | `ratio3_offset0p1` | 0.261 | 8.834 | 3.123 | 0.274 | 0.001004 | 0.000000 | 0.382 | 0.279 |
| 50k | `ratio3_offset0p2` | 0.254 | 9.244 | 3.197 | 0.260 | 0.000624 | 0.000000 | 0.339 | 0.259 |
| 100k | `ratio2_offset0p1` | 0.295 | 8.961 | 3.137 | 0.302 | 0.000244 | 0.000116 | 0.288 | 0.242 |
| 100k | `ratio2_offset0p2` | 0.217 | 9.305 | 3.190 | 0.245 | 0.000000 | 0.000000 | 0.337 | 0.278 |
| 100k | `ratio3_offset0p1` | 0.236 | 9.115 | 3.170 | 0.252 | 0.000109 | 0.000000 | 0.271 | 0.227 |
| 100k | `ratio3_offset0p2` | 0.242 | 9.423 | 3.216 | 0.256 | 0.000163 | 0.000000 | 0.275 | 0.216 |
| 150k | `ratio2_offset0p1` | 0.237 | 9.783 | 3.275 | 0.253 | 0.000000 | 0.000000 | 0.290 | 0.228 |
| 150k | `ratio2_offset0p2` | 0.267 | 8.866 | 3.106 | 0.283 | 0.000000 | 0.000000 | 0.299 | 0.233 |
| 150k | `ratio3_offset0p1` | 0.301 | 8.473 | 3.054 | 0.306 | 0.000271 | 0.000086 | 0.277 | **0.191** |
| 150k | `ratio3_offset0p2` | 0.237 | 9.664 | 3.248 | 0.254 | 0.000353 | 0.000000 | **0.269** | 0.213 |
| 200k | `ratio2_offset0p1` | **0.193** | **10.110** | **3.319** | **0.220** | 0.000244 | 0.000000 | 0.327 | 0.267 |
| 200k | `ratio2_offset0p2` | 0.209 | 9.727 | 3.249 | 0.244 | 0.000163 | 0.000000 | 0.311 | 0.256 |
| 200k | `ratio3_offset0p1` | 0.221 | 9.488 | 3.222 | 0.247 | **0.000081** | 0.000000 | 0.293 | **0.220** |
| 200k | `ratio3_offset0p2` | **0.193** | 9.979 | 3.292 | 0.223 | 0.000163 | 0.000000 | **0.274** | 0.221 |

`ratio3_offset0p2` and `ratio2_offset0p1` both improve effective alphabet
monotonically. Only `ratio3_offset0p2` combines that trend with a stable
geometry plateau after 100k. From 150k to 200k it gains 0.314 effective amino
acids while all-atom clashes increase only 0.005 and side-chain clashes
increase 0.008 per residue. By contrast, `ratio2_offset0p1` gains 0.327
effective amino acids over the same interval while all-atom and side-chain
clashes increase 0.037 and 0.039.

At fixed offset 0.2, ratio 3 beats ratio 2 at 200k on alanine fraction,
effective alphabet, all-atom clashes, and side-chain clashes. At fixed ratio
3, offset 0.2 strongly improves sequence diversity over offset 0.1 and lowers
all-atom clashes, with essentially unchanged side-chain clashes. This makes
`ratio3_offset0p2` the first-wave choice.

## Length-128 behavior at 200k

Length 128 remains the hardest slice, but none of the four compact cells shows
the severe alanine collapse of the old no-delay control.

| Arm | Alanine | Effective alphabet | Entropy | Max fraction | All-atom clashes/res. | Side-chain clashes/res. |
|---|---:|---:|---:|---:|---:|---:|
| `ratio2_offset0p1` | **0.241** | **9.783** | **3.270** | **0.250** | 0.317 | 0.262 |
| `ratio2_offset0p2` | 0.282 | 9.022 | 3.132 | 0.293 | 0.305 | 0.255 |
| `ratio3_offset0p1` | 0.299 | 8.609 | 3.078 | 0.304 | 0.269 | 0.209 |
| `ratio3_offset0p2` | 0.258 | 9.167 | 3.159 | 0.271 | **0.244** | **0.198** |

The length-128 result sharpens the same tradeoff: `ratio2_offset0p1` is the
sequence choice, whereas `ratio3_offset0p2` is the geometry-balanced choice.

## Expanded-wave terminal results

The 20-cell second wave currently has one complete 200k panel. Values are
means over 96 samples per row.

| Arm | Alanine | Effective alphabet | Entropy | Max fraction | CA clashes/res. | CA-CB bad | All-atom clashes/res. | Side-chain clashes/res. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ratio1_control` | 0.220 | 9.322 | 3.191 | 0.251 | 0.000407 | 0.000000 | 0.294 | 0.238 |
| `ratio1p5_offset0p05` | 0.249 | 9.330 | 3.190 | 0.265 | 0.000217 | 0.000000 | 0.300 | 0.234 |
| `ratio1p5_offset0p4` | 0.173 | 10.263 | 3.348 | 0.215 | 0.000163 | 0.000000 | 0.315 | 0.255 |
| `ratio2_offset0p05` | 0.226 | 9.614 | 3.239 | 0.248 | 0.000081 | 0.000000 | 0.328 | 0.241 |
| `ratio2_offset0p1_cbdelayed` | 0.263 | 9.009 | 3.140 | 0.275 | 0.000163 | 0.000000 | 0.305 | 0.236 |
| `ratio2_offset0p1_onset20` | 0.227 | 9.597 | 3.238 | 0.252 | 0.000434 | 0.000000 | 0.301 | 0.235 |
| `ratio2_offset0p1_onset5` | 0.259 | 9.250 | 3.177 | 0.278 | 0.000597 | 0.000000 | 0.297 | 0.237 |
| `ratio2_offset0p2_cbdelayed` | 0.168 | 10.372 | 3.363 | 0.211 | 0.000271 | 0.000000 | 0.323 | 0.262 |
| `ratio2_offset0p4` | 0.213 | 9.883 | 3.279 | 0.241 | 0.000732 | 0.000000 | 0.272 | 0.220 |
| `ratio3_offset0p05` | 0.208 | 9.758 | 3.266 | 0.236 | 0.000000 | 0.000000 | 0.292 | 0.221 |
| `ratio3_offset0p1_both` | 0.009 | 7.578 | 2.910 | 0.265 | 0.000326 | 0.032361 | 0.445 | 0.416 |
| `ratio3_offset0p1_cbdelayed` | 0.223 | 9.850 | 3.277 | 0.238 | 0.000000 | 0.000000 | 0.281 | 0.230 |
| `ratio3_offset0p1_lddt` | 0.127 | **10.854** | **3.431** | 0.186 | 0.000326 | 0.000000 | 0.340 | 0.307 |
| `ratio3_offset0p1_lddt_narrow` | 0.123 | 10.704 | 3.413 | **0.184** | 0.000244 | 0.000000 | 0.366 | 0.335 |
| `ratio3_offset0p1_lddt_wide` | 0.102 | 10.590 | 3.399 | 0.195 | 0.000163 | 0.000000 | 0.517 | 0.484 |
| `ratio3_offset0p1_onset20` | 0.193 | 9.965 | 3.296 | 0.237 | 0.000000 | 0.000000 | 0.316 | 0.272 |
| `ratio3_offset0p1_onset5` | 0.266 | 9.159 | 3.153 | 0.281 | 0.000163 | 0.000000 | 0.286 | 0.220 |
| `ratio3_offset0p1_selfcond` | 0.250 | 9.384 | 3.208 | 0.266 | 0.000000 | 0.000000 | 0.289 | 0.216 |
| `ratio3_offset0p2_cbdelayed` | 0.242 | 9.127 | 3.165 | 0.262 | 0.000488 | 0.000000 | 0.287 | 0.225 |
| `ratio4_offset0p05` | 0.273 | 8.909 | 3.114 | 0.289 | 0.000000 | 0.000000 | **0.268** | **0.203** |

## Factorial interpretation at 200k

### Clock surface

The useful ungated surface is broad rather than sharply peaked.
`ratio3_offset0p2` and `ratio2_offset0p4` are nearly matched geometry-balanced
points: effective alphabets 9.979 and 9.883, all-atom clashes 0.274 and 0.272,
and side-chain clashes 0.221 and 0.220. `ratio1p5_offset0p4` moves toward
sequence diversity (10.263) at a side-chain clash cost (0.255), while
`ratio4_offset0p05` moves toward geometry (0.203) at a large diversity cost
(8.909). Increasing delay strength alone is therefore not a general win.

### Cbeta membership

Cbeta membership interacts with ratio and offset rather than producing one
uniform main effect. The clearest matched pair is ratio 3/offset 0.2:
retaining Cbeta on the scaffold beats delaying it on alanine fraction (0.193
versus 0.242), effective alphabet (9.979 versus 9.127), all-atom clashes
(0.274 versus 0.287), and side-chain clashes (0.221 versus 0.225). Other pairs
trade sequence against geometry. The compact-clock evidence therefore favors
scaffold Cbeta for the promoted reference, without claiming a universal Cbeta
effect.

### Clock-gated objectives

Against the matched ungated `ratio3_offset0p1` scaffold control, the default
lDDT gate raises effective alphabet from 9.488 to 10.854 and lowers alanine
from 0.221 to 0.127, but raises all-atom clashes from 0.293 to 0.340 and
side-chain clashes from 0.220 to 0.307. The default gate has higher effective
alphabet and entropy and lower clash proxies than the narrower and wider
readiness windows, although those windows have slightly lower alanine.

The self-conditioning gate changes all-atom and side-chain clashes only from
0.293/0.220 to 0.289/0.216 while reducing effective alphabet to 9.384. The
combined gate fails far beyond an additive tradeoff and should not be reused
without diagnosing the objective interaction.

### Onset

For ratio 3/offset 0.1, onset 20 A moves toward diversity and away from
geometry, onset 5 A moves slightly toward geometry and away from diversity,
and the fixed 10 A onset remains the balanced point. The ratio-2 comparison
has the same diversity-versus-geometry structure. There is no evidence here
that moving onset away from 10 A resolves the central tradeoff.

## Comparison with the power-clock reference

| Arm and step | Alanine | Effective alphabet | Entropy | All-atom clashes/res. | Side-chain clashes/res. |
|---|---:|---:|---:|---:|---:|
| Legacy `k8_cb_scaffold`, 150k | 0.148 | 10.202 | 3.345 | **0.262** | **0.183** |
| Legacy `k8_cb_scaffold`, 200k | **0.117** | 10.442 | 3.376 | 0.317 | 0.216 |
| Compact `ratio3_offset0p2`, 150k | 0.237 | 9.664 | 3.248 | 0.269 | 0.213 |
| Compact `ratio3_offset0p2`, 200k | 0.193 | 9.979 | 3.292 | 0.274 | 0.221 |
| Compact `ratio2_offset0p4`, 200k | 0.213 | 9.883 | 3.279 | 0.272 | 0.220 |
| Compact `ratio3_offset0p1_lddt`, 200k | 0.127 | **10.854** | **3.431** | 0.340 | 0.307 |

The legacy 150k power-clock checkpoint has better geometry and at least
comparable diversity to most ungated compact points. Two ungated cells exceed
its effective alphabet: `ratio1p5_offset0p4` by 0.061 and
`ratio2_offset0p2_cbdelayed` by 0.170, but both have substantially worse clash
proxies. The lDDT-gated compact point improves sequence diversity further but
also worsens geometry. The offset formulation therefore maps the tradeoff
cleanly, but it has not yet produced a new overall winner. The next side-chain
improvement should target the low-noise objective or local chemical geometry
rather than widening the clock-only sweep.

## Provenance and execution ledger

The authoritative first-wave recovery root is:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
delayed-sidechain-offset-4x200k-recovery/
b5370102929d9f337a6e95b614745a01ef32298d/
c980642c52668a9bbab895578f4b0dd53cc08525/attempt2/v1
```

Workflow `hk-delayed-sidechain-offset-recovery-b537010-c980642-attempt2-v1`
has four succeeded trainers, sixteen succeeded samplers, and four succeeded
analyses. Missing legacy artifact events were repaired by the idempotent
publisher at commit `2ac5f36a8149de840121feb381e87decb9c357cf`; no sampler
or training task was duplicated. Each milestone contains 384 PDB files.

| Step | Analysis job | Analysis JSON SHA-256 |
|---:|---|---|
| 50k | `job-a00761bba1e81d5323c6` | `d035946e6801c9ab26b47286d743de877105c941633f95907111595b219f5cd6` |
| 100k | `job-023790e77eeb872e3f84` | `43babe0d103520ad7a127cc0e3e3e09d9272e71745192d2f80543ff697302df2` |
| 150k | `job-e199661b316bc8bae164` | `e09c50b61e7e297230f6f46aa7d3d6afd8d18ee0ea691b7cab1cf80c8e9cfe33` |
| 200k | `job-4a548b94b15c331cbc5f` | `f53515cf6257be7ae00f529e3695075cda9aba2d2f7914bc93ed7075517b7a7d` |

| First-wave arm | 200k checkpoint SHA-256 |
|---|---|
| `ratio2_offset0p1` | `c423a674b44ca3472a91282322e22a6c9d3b22549a8802f51ab553f161507554` |
| `ratio2_offset0p2` | `b92837ff621b4aa4dbbadf759c50c11ebbd63519ed535cabdebb49a6b247a32a` |
| `ratio3_offset0p1` | `a585e6a7c4e899a3a33ea593975f7f0977831d71bf1a3fc9edb172526d7b078d` |
| `ratio3_offset0p2` | `84e16da2f04e63b8b322c7804e6dee25cfbda4f58383b5bd3c3ccb3bf4f710b8` |

The authoritative second-wave root is:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
delayed-sidechain-offset-screen20/
1d26d42c96d61bc9ed218e1b8a004f3712afc988/v1
```

Workflow `hk-delayed-sidechain-offset-screen20-1d26d42-v1` produced 1,920
PDB files at 200k. Analysis job `job-c61498201cd475087210` succeeded; its JSON
SHA-256 is
`3abae477c076e876dd7eee1e1a71f28da9afa144484149b6a174971d4b9534b6`.
Both waves use pinned Koochak commit
`d186e7cc1533446165e5a92d504f2c5a0c051409`.
