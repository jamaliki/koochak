# Experiment ledger

This directory records model, training, sampling, and evaluation experiments.
Entries should separate measured results from hypotheses, identify immutable
commits and campaign roots, and state whether numbers are diagnostics, point
estimates, or confidence intervals.

## Canonical evaluation rules

- ESMFold mean pLDDT must come from actual Atom37 PDB `ATOM` records. A current
  summary identifies this as `pdb_atom37_masked_b_factors`.
- Historical pLDDT and strict-designability values that averaged zero-filled
  Atom37 slots are stale. CA RMSD, generated structures, sequences, and clash
  metrics were unaffected by that confidence-only bug.
- Training completion, sample generation, ESMFold folding, and aggregate
  analysis are separate milestones.
- Tiny deterministic panels are useful activation and geometry diagnostics,
  not designability estimates or confidence intervals.

## Current synthesis and cross-campaign entries

The durable end-to-end Atom4 plus residue-latent record is maintained in the
[experimental-ledger directory](../experimental-ledger/README.md), including
the exact codec objectives, architecture contract, continuation tables, and
provenance for completed and incomplete stages.

| Entry | Status | Main conclusion |
| --- | --- | --- |
| [Canonical ESMFold designability rescore](esmfold_designability_rescore_20260824.md) | Complete | Corrected pLDDT changes model and sampler conclusions; mature Kaveh is often MPNN-saveable but still trails released Pallatom in raw co-design |
| [Old-Kaveh training factorial rescore](old_kaveh_training_factorial_18x200k_rescore.md) | Complete | The recovered old structural sampler, not one training trick, explains the largest strict-designability gain |
| [Low-noise inverse-folding training and refinement](inverse_folding_training_and_refinement_20260825.md) | Evaluation incomplete | Five of six 400k trajectories completed; one storage failure prevents a valid six-arm ranking |
| [Signal propagation and recycling redesign](signal_propagation_and_recycling_20260825.md) | Active | Residual gauge, pair geometry, Q/K, and atom recurrence fixes are validated; adaptive modulation and final 400k quality remain open |
| [Recycling, sampling, and geometry lessons](recycling_sampling_and_geometry_lessons_20260825.md) | Mixed | Records the quadrature alignment bug, oxygen clash attribution, AF2 clip30 result, steric status, and length-bucket evidence gap |
| [Hierarchy recycling throughput probe](hierarchy_recycling_throughput.md) | Complete engineering screen | All-stage recycling costs about 12% p50; intermediate supervision adds a smaller cost and little memory |
| [Fixed-sampler feature architecture screen](fixed_sampler_feature_architecture_screen.md) | Complete | Compares feature architectures under one controlled sampler |
| [Filtered all-16 ESMFold recovery v2](filtered_all16_esmfold_recovery_v2_20260830.md) | Partial endpoint | Corrected replay has 18 completed panels; early pLDDT is modest and strongly length-dependent |
| [Latent recovery monitor 2026-08-30 06:22](latent_recovery_monitor_20260830_0622.md) | Active | b256 replacement training has verified 250k and 50k checkpoints; no active workflow failures |
| [Latent recovery monitor 2026-08-30 07:22](latent_recovery_monitor_20260830_0722.md) | Active | Queue is healthy; an unrelated active 4-GPU run is ahead of latent jobs, with no active latent failures |

## Delayed side-chain diffusion

| Entry | Scope |
| --- | --- |
| [Five-arm delayed side-chain 200k results](delayed_sidechain_5x200k_results.md) | Initial delayed-side-chain comparison |
| [Delayed-sidechain offset 24x200k](delayed_sidechain_offset_24x200k_results.md) | Offset/clock factorial |
| [Length-128 offset ESMFold](delayed_sidechain_offset_length128_esmfold.md) | L128 foldability endpoint; use its corrected sections where present |
| [Best-eight length-256 plan/results](delayed_sidechain_offset_best8_length256_200k.md) | L256 continuation; cross-check actual completed artifacts before citing |

## Coarse transport and local-center architecture

| Entry | Scope |
| --- | --- |
| [Coarse signal transport](coarse_signal_transport_20x100k_results.md) | Initial transport study |
| [Filtered coarse signal transport](coarse_signal_transport_filtered_20x100k_results.md) | Filtered replication |
| [Local-center architecture 7x100k](local_center_architecture_7x100k_results.md) | Core local-center factorial |
| [Local-center coarse continuation](local_center_coarse_continuation_6x200k_results.md) | Longer coarse comparison |
| [Coordinate alignment](local_center_coordinate_alignment_1x100k_results.md) | Alignment ablation |
| [Decoder follow-up](local_center_decoder_followup_8x100k_results.md) | Decoder variants |
| [Distogram at 100k](local_center_distogram_100k.md) | Distogram behavior |
| [Filtered control at 200k](local_center_filtered_control_200k_results.md) | Filtered control continuation |
| [Secondary structure 4x100k](local_center_secondary_structure_4x100k_results.md) | Secondary-structure features |
| [Sequence diversity 16x100k plan](local_center_sequence_diversity_16x100k.md) | Sequence-diversity design |
| [Sequence diversity 16x100k results](local_center_sequence_diversity_16x100k_results.md) | Sequence-diversity outcomes |
| [Short L128 100k screen](short128_100k_screen.md) | Early L128 screen |

## Entry checklist

New ledger entries should include, where applicable:

1. question and decision summary;
2. exact arms, data distribution, optimizer, batch, and sampler;
3. immutable code commit, workflow ID, campaign root, and checkpoint maturity;
4. failures and skipped dependents, including whether they are model or
   infrastructure failures;
5. point estimates plus matched/bootstrap uncertainty when available;
6. canonical pLDDT provenance;
7. explicit statements of what is and is not established;
8. the next experiment required to change the decision.
