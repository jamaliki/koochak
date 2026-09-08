# Architecture factorial results: DiT, objective, and stability endpoints (2026-09-08)

## Scope and endpoint

This entry records newly landed durable outputs from the L128 factorials. Unless
noted otherwise, counts are out of the same 32-sample panel and use strict
designability:

```text
CA Kabsch RMSD < 2 A AND mean ESMFold pLDDT > 80
```

The primary diversity count is the number of complete-linkage Progres clusters
among the ESMFold-designable samples. These are structural self-consistency
surrogates, not experimental folding measurements.

## DiT counterparts: 250k result

The DiT AdaLN-zero counterpart workflow has now produced 250k analyses for all
four completed architecture cells:

| Arm | Designable at 250k | Progres clusters | Sequence identity mean |
|---|---:|---:|---:|
| `flat_after_node_no_transition-depth-attn-sandwich` | **17/32** | **6** | 0.100 |
| `flat_after_node_no_transition-depth-attn-no-sandwich` | 4/32 | 3 | 0.091 |
| `pool_before_attention_pair_transition-depth-attn-no-sandwich` | 7/32 | 3 | 0.089 |
| `flat_after_node_no_transition-full-attn-sandwich` | 0/32 | 0 | 0.655 |

Interpretation: the depth-scaled DiT arm with sandwich normalization remains the
clear DiT winner. The no-sandwich flat-depth arm has continued its late decline
(17 at 100k, 11 at 150k, 5 at 200k, 4 at 250k), so this is evidence of drift,
not merely a noisy early checkpoint. Full attention remains a negative control:
its high sequence identity and zero designability are consistent with collapse.

Durable source root:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/dit-adaln-zero-counterparts-L128/142664f5526bb8f14da6f0570b25598be5e8d4cc
```

## Objective decomposition: 50k to 250k

The 50k objective ablation was complete for all ten cells. The continuation
workflow has now produced all ten LDDT cells at 250k, plus `coordseq-atom14`
through 400k:

| Cell | 50k designable | 250k designable | 250k clusters |
|---|---:|---:|---:|
| `lddt-m0g0c0` | 2 | 11 | 6 |
| `lddt-m0g0c1` | 0 | 24 | 3 |
| `lddt-m0g1c0` | 2 | 25 | 3 |
| `lddt-m0g1c1` | 6 | 27 | 3 |
| `lddt-m1g0c0` | 8 | 24 | 3 |
| `lddt-m1g0c1` | 11 | **29** | 4 |
| `lddt-m1g1c0` | 2 | 17 | 6 |
| `lddt-m1g1c1` | 11 | 26 | 4 |
| `coordseq-atom14` | 1 | 15 | 3 |
| `coordseq-ca` | 4 | 2 | 2 |

The matched 250k comparison suggests that inverse `c_out` compensation is the
strongest single objective-axis signal in this batch: mean designable count is
26.5 with compensation versus 19.25 without it across the eight LDDT cells.
The effect is not monotone across the gate or mask axes, so this is a useful
factorial lead rather than a final causal claim. The best current objective
cell is `lddt-m1g0c1` at 29/32. No LDDT 500k result has landed yet; the two
continuation workflows still have training and downstream analysis work
remaining.

Durable source root:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/objective-decomp-continuations-500k-L128/0db43cf38f3a8d9569044a30e5731d81af661536
```

## Non-DiT stability factorial: late endpoint evidence

The stability factorial is not a clean matched 500k table because several
downstream jobs are blocked or absent. The durable trajectory nonetheless
shows a major stability concern:

- `flat_after_node_no_transition-depth-attn-no-sandwich`: 19/32 at 300k,
  with no later durable analysis.
- `pool_before_attention_pair_transition-depth-attn-no-sandwich`: 8/32 at
  500k, after 26/32 at 450k.
- `flat_after_node_no_transition-depth-attn-sandwich`: 0/32 at 500k.
- `flat_after_node_no_transition-full-attn-sandwich`: 0/32 at 500k.
- Recovery probe `flat-strict-sc0p5`: 5/32 at 500k.

The 26-to-8 drop in the pooled depth/no-sandwich arm is the strongest newly
landed non-monotonicity signal. It argues against using a single intermediate
checkpoint as evidence of stable training and is a direct reason to prioritize
the gain-invariant diagnostics. The zero results for the non-DiT sandwich arm
should not be interpreted as proof that sandwich normalization is intrinsically
bad: the relevant cell has an incomplete/failed campaign history and is not
the same as the successful DiT counterpart.

Durable source root:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/transformer-stability-factorial-500k-L128/71cd8fdfed7f826b8a2ebe7a7dc4c28f31984e90
```

## What is and is not new experimental evidence

- The corrected gain-invariant factorial (`691f07a`) has passed its preflights
  and is training 12 cells, but has no scientific checkpoint result yet.
- The separately requested pooled-depth+sandwich DiT job (`37598e7`) is still
  training; no analysis artifact has landed.
- The earlier failed gain-invariant launch (`9955962`) is an infrastructure
  failure from the bounded-DiT MLP path and must not enter any scientific
  comparison.

## Current working conclusion

For production candidates, retain the DiT
`flat_after_node_no_transition-depth-attn-sandwich` arm as the current baseline.
Do not promote no-sandwich or full-attention variants on the basis of an early
peak. The next decisive evidence is the gain-invariant factorial, with special
attention to per-layer Q/K RMS, pair scaled-delta/stream ratios, raw branch
updates before normalization, and whether the 250k DiT ranking survives under
the new parameter-free gain package.
