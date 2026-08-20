# Delayed-sidechain power-clock factorial: 50k-200k results

## Scope and decision summary

This entry closes the five-arm Cartesian delayed-sidechain campaign. The fixed
control is the filtered `3/3/8/3/3` local-center model with no Kabsch loss,
polar weight 2, sequence sigma ramp 0.5 to 1, and the validated length/pLDDT/
loop/packing filters. The factorial compares no delay with power-clock delays
at kappa 4 and 8, with Cbeta either retained on the backbone clock
(`cb_scaffold`, first delayed Atom14 slot 5) or delayed with the side chain
(`cb_delayed`, first delayed slot 4).

The decision depends on the downstream objective:

* **Carry `k8_cb_scaffold` forward.** At 200k it has the best sequence panel:
  alanine fraction 0.117, effective alphabet 10.442, entropy 3.376 bits, and
  maximum-residue fraction 0.193. Its CA-step failure fraction is zero and its
  CA-clash proxy remains very low.
* **Keep both its 150k and 200k checkpoints.** The 200k checkpoint is the
  sequence-diversity choice. The 150k checkpoint has better all-atom and
  side-chain clash proxies with nearly the same effective alphabet.
* **Keep Cbeta on the backbone clock.** Delaying Cbeta increases CA-CB bond
  failures and reduces sequence diversity, especially at kappa 8.
* **Do not promote `delay_off`.** Its 200k length-128 panel is strongly
  alanine-biased (0.459) with effective alphabet 6.130 despite acceptable
  backbone proxies.

These are fixed-panel, single-training-seed results. Clash counts and bond
thresholds are diagnostic geometry proxies, not chemistry-qualified structure
validation or foldability evidence.

The subsequent 24-arm compact offset-clock campaign is recorded in
[Delayed-sidechain compact offset clock: 24-arm results](delayed_sidechain_offset_24x200k_results.md).

## Fixed training and sampling contract

All arms use architecture `3/3/8/3/3`, local-center pair geometry, coordinate
self-conditioning, no Kabsch coordinate alignment, batch size 32, BF16,
Adam at `3e-4`, EMA decay 0.999, and compiled training. Dataset filters are
minimum length 32, maximum length 128, mean pLDDT greater than 80, loop length
less than 15, loop content less than 0.4, and packing density greater than 0.3.

Every milestone contains 96 EMA samples: 32 each at lengths 64, 96, and 128.
Sampling uses seed `20260820`, BF16, 200 uncompiled Euler steps, step scale
2.25, churn 0.2, and the uniform probability grid. The EDM rho-5/sigma-0.003
fields are present in the continuation config but inactive because
`time_grid_mode=uniform`.

Training began at source commit
`7728fea843bfb8b0c452738749ef678faf56ec7f`. The cancelled trainers were
resumed with full model, optimizer, EMA, RNG, and step state at code commit
`3a4bfe8b0215ef80a36fbcbec06f19bd19fc7a56`: `delay_off` from 180k and the
four delayed arms from 190k. The pinned Koochak commit is
`d186e7cc1533446165e5a92d504f2c5a0c051409`.

## Aggregate milestone results

Values are means over 96 samples. Higher effective alphabet and entropy and
lower alanine/max-residue fractions are preferable. Lower clash and bond-bad
fractions are preferable.

| Step | Arm | Alanine | Effective alphabet | Entropy | Max fraction | CA clashes/res. | CA-CB bad | All-atom clashes/res. | Side-chain clashes/res. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | `delay_off` | 0.349 | 7.859 | 2.934 | 0.357 | 0.001356 | 0.000000 | 0.400 | 0.302 |
| 50k | `k4_cb_scaffold` | 0.218 | 9.809 | 3.284 | 0.230 | 0.000949 | 0.000250 | 0.433 | 0.321 |
| 50k | `k4_cb_delayed` | 0.230 | 9.749 | 3.277 | 0.241 | 0.001058 | 0.003035 | 0.419 | 0.308 |
| 50k | `k8_cb_scaffold` | 0.159 | 9.883 | 3.298 | 0.211 | 0.001058 | 0.000280 | 0.294 | 0.157 |
| 50k | `k8_cb_delayed` | 0.184 | 10.031 | 3.321 | 0.205 | 0.000732 | 0.013165 | 0.361 | 0.249 |
| 100k | `delay_off` | 0.351 | 7.862 | 2.922 | 0.357 | 0.000543 | 0.000000 | 0.291 | 0.223 |
| 100k | `k4_cb_scaffold` | 0.216 | 9.719 | 3.268 | 0.232 | 0.000678 | 0.000284 | 0.354 | 0.250 |
| 100k | `k4_cb_delayed` | 0.239 | 9.372 | 3.214 | 0.253 | 0.000217 | 0.001129 | 0.314 | 0.225 |
| 100k | `k8_cb_scaffold` | 0.147 | 10.181 | 3.341 | 0.200 | 0.000190 | 0.000000 | 0.271 | 0.175 |
| 100k | `k8_cb_delayed` | 0.190 | 10.265 | 3.353 | 0.214 | 0.000190 | 0.006283 | 0.315 | 0.233 |
| 150k | `delay_off` | 0.311 | 8.436 | 3.017 | 0.320 | 0.000326 | 0.000000 | 0.295 | 0.205 |
| 150k | `k4_cb_scaffold` | 0.240 | 9.520 | 3.231 | 0.257 | 0.001112 | 0.000000 | 0.351 | 0.253 |
| 150k | `k4_cb_delayed` | 0.292 | 8.770 | 3.100 | 0.303 | 0.000515 | 0.001674 | 0.285 | 0.196 |
| 150k | `k8_cb_scaffold` | 0.148 | 10.202 | 3.345 | 0.204 | 0.000570 | 0.000000 | **0.262** | **0.183** |
| 150k | `k8_cb_delayed` | 0.211 | 9.860 | 3.288 | 0.233 | 0.000271 | 0.003792 | 0.278 | 0.195 |
| 200k | `delay_off` | 0.339 | 7.855 | 2.895 | 0.349 | 0.000163 | 0.000000 | 0.295 | 0.201 |
| 200k | `k4_cb_scaffold` | 0.241 | 9.568 | 3.238 | 0.261 | 0.000217 | 0.000000 | 0.321 | 0.237 |
| 200k | `k4_cb_delayed` | 0.247 | 9.338 | 3.191 | 0.265 | 0.000515 | 0.000846 | 0.342 | 0.225 |
| 200k | `k8_cb_scaffold` | **0.117** | **10.442** | **3.376** | **0.193** | 0.000244 | 0.000171 | 0.317 | 0.216 |
| 200k | `k8_cb_delayed` | 0.208 | 9.957 | 3.302 | 0.232 | 0.000190 | 0.002751 | 0.303 | 0.217 |

All arms and milestones have zero CA-step bad fraction.

## Main effects at 200k

### Delay strength

At fixed scaffold Cbeta, kappa 8 dominates kappa 4 on sequence diversity:
effective alphabet increases from 9.568 to 10.442, alanine fraction falls from
0.241 to 0.117, and maximum-residue fraction falls from 0.261 to 0.193. It also
slightly improves the all-atom and side-chain clash proxies. Relative to
`delay_off`, `k8_cb_scaffold` gains 2.587 effective amino acids and 0.481 bits
of entropy while reducing alanine fraction by 0.222.

### Cbeta clock membership

At kappa 8, retaining Cbeta on the scaffold improves effective alphabet by
0.485 and lowers alanine fraction by 0.090 relative to delaying Cbeta. The
CA-CB bad fraction is 0.000171 versus 0.002751, approximately a 16-fold
difference. The kappa-4 comparison has the same direction for sequence and
CA-CB integrity, although its side-chain clash proxy is slightly lower when
Cbeta is delayed. The combined evidence favors `first_delayed_slot=5`.

### Training horizon

For `k8_cb_scaffold`, 150k to 200k improves effective alphabet from 10.202 to
10.442 and reduces alanine fraction from 0.148 to 0.117. This comes with an
all-atom clash increase from 0.262 to 0.317 (20.7%) and a side-chain clash
increase from 0.183 to 0.216 (17.9%). The 150k checkpoint is therefore the
balanced geometry choice; 200k is the sequence-diversity choice. This
divergence motivates the compact offset clock and clock-gated objective screen
rather than simply training the power clock longer.

## Length dependence at 200k

| Arm | Length | Alanine | Effective alphabet | Entropy | Max fraction | CA clashes/res. | All-atom clashes/res. |
|---|---:|---:|---:|---:|---:|---:|---:|
| `delay_off` | 64 | 0.197 | 10.188 | 3.325 | 0.225 | 0.000000 | 0.267 |
| `delay_off` | 96 | 0.362 | 7.247 | 2.810 | 0.362 | 0.000000 | 0.320 |
| `delay_off` | 128 | 0.459 | 6.130 | 2.550 | 0.461 | 0.000488 | 0.297 |
| `k8_cb_scaffold` | 64 | 0.099 | 10.472 | 3.380 | 0.197 | 0.000000 | 0.275 |
| `k8_cb_scaffold` | 96 | 0.115 | 10.200 | 3.340 | 0.194 | 0.000000 | 0.329 |
| `k8_cb_scaffold` | 128 | 0.139 | 10.654 | 3.407 | 0.188 | 0.000732 | 0.346 |

The delay benefit is strongest at length 128: the no-delay arm collapses
toward alanine while `k8_cb_scaffold` retains the most diverse sequence panel
of its three lengths. Backbone proxies remain clean enough that ESMFold or a
comparable folding screen, not these heuristics, should determine which
sequences are genuinely foldable.

## Provenance and execution ledger

The authoritative continuation root is:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
delayed-sidechain-5x200k-continuation/
7728fea843bfb8b0c452738749ef678faf56ec7f/
3a4bfe8b0215ef80a36fbcbec06f19bd19fc7a56/v1/
```

| Arm | 200k checkpoint SHA-256 | Trainer job | Sampler job |
|---|---|---|---|
| `delay_off` | `ced9153739c6a061e9456609c0f335175d9bb2ac36fdedad0c981380abd0f085` | `job-f5f99c911924be640993` | `job-d7706cf85abaac4c423c` |
| `k4_cb_scaffold` | `d0b35295a3c79588ed3ccdf88aab9a5be3292c632ec73aeb71fed504978c84d1` | `job-1e18cb84d0a391c933be` | `job-7f2515768ad9c6091e9e` |
| `k4_cb_delayed` | `77a787de6c64149b499044b8af0aafc9afe2544ef08f34e322daff650c3180ae` | `job-1d9580da5ba98e0a6a07` | `job-54bbe739795cd3ea5efb` |
| `k8_cb_scaffold` | `db9eebb43f4659118371a6c9db993f2eaeb5805b69945cfe16083770d9222830` | `job-ef21d947ee5ce6ed0770` | `job-a96fc4e5a8d1a44d103a` |
| `k8_cb_delayed` | `353aa84cf5d3f584d542a95a749a87805dc561dfbef6bd42c6625c11b7d62236` | `job-28e2265297f5482b90fc` | `job-880ec18584d205d8b9fb` |

All five trainers, five samplers, and analysis job
`job-28ee5b43a49761360e54` succeeded. The continuation's terminal artifact
notifications were reconciled idempotently by five CPU publisher jobs; no
sampler was duplicated. The authoritative 200k analysis is
`analysis/milestone_step200000.json` under the continuation root. The 50k,
100k, and 150k analyses remain under the original source campaign root.
