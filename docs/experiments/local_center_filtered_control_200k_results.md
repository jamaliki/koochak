# Filtered local-center control: 50k-200k results

## Scope and decision summary

This entry records the single-seed continuation of the selected local-center
control through 200,000 training steps. The model is the `3/3/8/3/3`
atom-encoder/residue-encoder/coarse/residue-decoder/atom-decoder control with
no Kabsch alignment loss, the `lin0510p2` sequence schedule, local-center pair
geometry, coordinate self-conditioning, and no intermediate distograms or
intermediate distogram feedback.

The 150k checkpoint is the best descriptive panel in this run: it has the
highest effective alphabet and entropy, lowest maximum-residue fraction, and
lowest CA-clash proxy. The 200k panel remains valid and geometrically clean,
but sequence diversity regresses relative to 150k. This is a checkpoint-screen
result, not evidence that longer training is universally harmful.

## Dataset and model contract

The resolved training and sampling manifests record the following data filter:

| Filter | Configured value | Applied semantics |
| --- | ---: | --- |
| Minimum length | 32 | inclusive |
| Maximum length | 128 | inclusive |
| Minimum mean pLDDT | 80.0 | strict: `> 80.0` |
| Maximum loop length | 15 | strict: `< 15` |
| Maximum loop content | 0.4 | strict: `< 0.4` |
| Minimum packing density | 0.3 | strict: `> 0.3` |

The loss configuration has coordinate weight 1.0, sequence weight 0.25,
smooth-LDDT weight 1.0, distogram weight 0.5, polar weight 2.0, zero
coordinate alignment loss, and zero marginal-JS and intermediate-distogram
loss. Training used seed 42, EMA decay 0.999, BF16, and compiled inference.

## Fixed sampling panel and provenance

Every milestone contains 96 EMA samples: 32 each at lengths 64, 96, and 128.
Sampling used seed `20260813`, batch size 32, BF16 compiled inference, and
the 200-step corrected sampler. The source commit is
`f6158b6a98aceab16a1b40b3285a7a166bc5f8ad`; the pinned Koochak commit is
`d186e7cc1533446165e5a92d504f2c5a0c051409`.

| Checkpoint | Checkpoint SHA-256 | Sample attempt | Analysis artifact |
| ---: | --- | ---: | --- |
| 50,000 | `cb92ecdfe4f17062e0a64f339467dce76186754775bcfde82b774a381c018900` | 04 | `analysis/step050000_attempt04/milestone_step050000.json` |
| 100,000 | `b8449dd91d78dd2c3e0f65ca5748e9da00f2b6cdf55284ec6c4f2504ab82a1fa` | 01 | `analysis/step100000_attempt01/milestone_step100000.json` |
| 150,000 | `3c141fbf62e177ce8dec25c1482e3aa6a3f92cfcfe63c75701efb0216db5d2f1` | 04 | `analysis/step150000_attempt04/milestone_step150000.json` |
| 200,000 | `e35c6183fb635c779bc513795910011a6545b72ae6879ef77ef77bc2ff385fce` | 04 | `analysis/step200000_attempt04/milestone_step200000.json` |

The authoritative remote run root is:

```text
/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/
local-center-filtered-control-200k/
f6158b6a98aceab16a1b40b3285a7a166bc5f8ad/
```

## Aggregate panel results

Values are means over the 96-sample panel. Higher effective alphabet and
entropy are preferable; lower maximum-residue fraction, homopolymer run, and
CA-clash proxy are preferable. CA clashes are a coarse geometry warning proxy,
not a chemistry-qualified structural validation.

| Checkpoint | Effective alphabet | Entropy (bits) | Max residue fraction | Max run | CA-step bad fraction | CA clashes/residue |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 7.885 | 2.927 | 0.354 | 4.781 | 0.000 | 0.001872 |
| 100k | 7.836 | 2.901 | 0.360 | 5.156 | 0.000 | 0.000597 |
| 150k | **8.236** | **2.984** | **0.338** | **4.448** | 0.000 | **0.000163** |
| 200k | 7.557 | 2.852 | 0.362 | 4.781 | 0.000 | 0.000515 |

## Length-stratified results

| Length | Checkpoint | Effective alphabet | Entropy | Max fraction | Max run | CA clashes/residue |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 50k | 8.603 | 3.076 | 0.320 | 3.750 | 0.000000 |
| 64 | 100k | 9.204 | 3.172 | 0.289 | 3.594 | 0.000000 |
| 64 | 150k | 9.494 | 3.212 | 0.264 | 3.375 | 0.000000 |
| 64 | 200k | **9.514** | **3.223** | **0.262** | **2.969** | 0.000000 |
| 96 | 50k | 8.357 | 3.015 | 0.315 | 4.250 | 0.000977 |
| 96 | 100k | 7.994 | 2.945 | 0.335 | 4.625 | 0.000326 |
| 96 | 150k | 7.923 | 2.923 | 0.351 | 4.781 | 0.000000 |
| 96 | 200k | 6.843 | 2.725 | 0.380 | 5.188 | 0.000326 |
| 128 | 50k | 6.695 | 2.690 | 0.425 | 6.344 | 0.004639 |
| 128 | 100k | 6.309 | 2.586 | 0.455 | 7.250 | 0.001465 |
| 128 | 150k | **7.293** | **2.816** | **0.399** | **5.188** | **0.000488** |
| 128 | 200k | 6.314 | 2.608 | 0.445 | 6.188 | 0.001221 |

The length dependence is strong. Length 64 improves monotonically through
200k, while length 128 remains the diversity and geometry bottleneck. The
150k checkpoint is the best length-128 snapshot as well as the best aggregate
checkpoint. Length 96 is stable through 150k but declines at 200k.

## Interpretation

- The 50k to 100k change is slightly negative for sequence diversity, while
  the 150k panel improves on both earlier checkpoints.
- From 150k to 200k, effective alphabet falls by 0.680 and entropy by 0.132;
  maximum-residue fraction rises by 0.024 and the CA-clash proxy rises by
  0.000353. The CA-step bad fraction remains zero.
- The 150k panel is the current checkpoint to carry forward for this control
  if the decision is based on this fixed sampling panel. The 200k checkpoint
  should remain archived as a valid continuation rather than being discarded.

These comparisons use different random initializations for each checkpoint
panel and one training seed, so the horizon deltas are descriptive rather
than paired training-seed estimates. The analysis does not include bootstrap
intervals or all-atom chemistry validation.

## Execution and recovery ledger

- Producer: Koochak/Scruffy job `job-6300540825a32cac5472`, workflow
  `hk-local-center-filtered-control-200k-f6158b6-v1`, task `train-control`,
  attempt 3. It published the four numbered checkpoints above.
- Samplers: 50k `job-b12f415f66b379d20a25`, 100k
  `job-07531cc6b1be82263929`, 150k `job-b8a46527f6db7248a8f1`, and 200k
  `job-1b6e9709b1f7169e22cf`; all four terminal records are `succeeded` with
  exit code 0. The 50k/150k/200k panels use recovery attempt 4; 100k uses
  attempt 1.
- Analyses: 50k `job-ee0c550bd07da1894fbe`, 100k
  `job-cd63299698f345227903`, and 200k `job-f5d0cc6de2df4c7057c2` have
  terminal `succeeded` records. The 150k analysis
  `job-78e60954e39704bbb9cc` wrote the complete JSON with exit code 0, but
  Scruffy later marked the job `runtime_placement_invalid`; the JSON was
  independently validated and is retained.
- Archives: 50k `job-e9e6d17d2a2f086532f1`, 200k
  `job-e6a646fe0176ddcec839`, and the 100k/150k archive jobs
  `job-0a1af7da50d9717759bb` and `job-db5cf76446d6d30d6c42` produced complete
  archives. The latter two were also marked `runtime_placement_invalid`
  after writing; all four archives were independently size- and SHA-256-
  validated before transfer.

## Local artifact copy

The verified local copies are under:

```text
/Users/kiarash.jamali/Documents/ChatGPT/hierarchical-kaveh/
artifacts/local-center-filtered-control-200k/
```

Each `step*_attempt*` directory contains `structures/` with 96 extracted PDB
files, `structures.tar`, `structures.json`, `sample_manifest.json`, and the
milestone analysis JSON. The four remote archive SHA-256 values were matched
exactly after chunked transport.
