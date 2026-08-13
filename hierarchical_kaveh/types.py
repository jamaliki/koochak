"""Tensor contracts shared by training, sampling, and the denoiser."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class DenoiserInput:
    """One rectangular, prefix-padded Atom14 denoising batch.

    Coordinates are in Angstroms and ``sigma`` is per Atom14 slot.
    """

    coordinates: Tensor
    sigma: Tensor
    residue_index: Tensor
    chain_index: Tensor
    chain_break: Tensor
    atom_mask: Tensor
    aatype_input: Tensor | None = None
    self_conditioned_coordinates: Tensor | None = None

    @property
    def residue_mask(self) -> Tensor:
        """Residues with a valid CA atom."""

        return self.atom_mask[..., 1].to(torch.bool)

    def to(self, device: torch.device | str) -> "DenoiserInput":
        """Move every tensor field to ``device``."""

        values = {
            item.name: (
                value.to(device=device) if isinstance(value := getattr(self, item.name), Tensor) else value
            )
            for item in fields(self)
        }
        return type(self)(**values)

    def with_self_conditioning(
        self,
        prediction: "Prediction | None",
    ) -> "DenoiserInput":
        """Return an input carrying a detached previous denoiser prediction."""

        if prediction is None:
            return replace(self, self_conditioned_coordinates=None)
        return replace(self, self_conditioned_coordinates=prediction.coordinates.detach())


@dataclass(frozen=True)
class CompactDistogram:
    """Patch-scale logits plus the p=4 map needed for residue-pair loss."""

    coarse_logits: Tensor
    slot_bias: Tensor
    residue_to_patch: Tensor
    residue_slot: Tensor
    patch_residue_index: Tensor
    residue_mask: Tensor
    symmetrize: bool = True

    def detach(self) -> "CompactDistogram":
        """Detach the differentiable tensors while preserving layout metadata."""

        return type(self)(
            coarse_logits=self.coarse_logits.detach(),
            slot_bias=self.slot_bias.detach(),
            residue_to_patch=self.residue_to_patch,
            residue_slot=self.residue_slot,
            patch_residue_index=self.patch_residue_index,
            residue_mask=self.residue_mask,
            symmetrize=self.symmetrize,
        )


@dataclass(frozen=True)
class Prediction:
    """The model's final coordinate, sequence, and residue-distance predictions."""

    coordinates: Tensor
    aatype_logits: Tensor
    distogram: Tensor | CompactDistogram | None = None

    def detach(self) -> "Prediction":
        """Detach a prediction for reuse as self-conditioning."""

        distogram = self.distogram
        if isinstance(distogram, Tensor):
            distogram = distogram.detach()
        elif isinstance(distogram, CompactDistogram):
            distogram = distogram.detach()
        return type(self)(
            coordinates=self.coordinates.detach(),
            aatype_logits=self.aatype_logits.detach(),
            distogram=distogram,
        )
