# Experimental ledger

This directory is the durable, answer-first record of the Atom4 plus residue-
latent experiments. Entries distinguish measured results from interpretation,
retain exact configuration contracts, and record missing or failed stages rather
than silently treating them as zeroes.

## Entries

- [Atom4 plus latent codec and diffusion campaign](atom4-latent-codec-and-diffusion.md)
- [Filtered latent recycling-best2 at 100k](filtered-recycling-best2-step100k.md)

The older, broad ESMFold rescore remains the canonical machine-readable
inventory for historical panels:

- [Canonical ESMFold designability rescore](../experiments/esmfold_designability_rescore_20260824.md)
- [Corrected panel inventory JSON](../experiments/artifacts/esmfold_designability_rescore_20260824.json)

## Common endpoint

Unless explicitly labeled otherwise, **strict designability** means:

```text
CA Kabsch RMSD < 2 A AND mean ESMFold pLDDT > 80
```

pLDDT is averaged over actual `ATOM` records in the predicted PDB, not all 37
possible Atom37 slots. All-atom comparisons use matched physical Atom14 heavy
atoms after CA alignment, with local terminal-symmetry corrections for ASP,
GLU, PHE, and TYR. These are ESMFold self-consistency surrogates, not
experimental folding measurements.
