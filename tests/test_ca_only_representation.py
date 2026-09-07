from dataclasses import replace
from pathlib import Path

import pytest
import torch

import hierarchical_kaveh.diffusion.losses as losses_module
from hierarchical_kaveh.config import LossConfig, ModelConfig, SamplingConfig
from hierarchical_kaveh.diffusion.corruption import corrupt_structure
from hierarchical_kaveh.io import write_sample_batch
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.sampling import build_topology, sample
from hierarchical_kaveh.types import DenoiserInput, Prediction


def _clean_structure(residues: int = 4) -> dict[str, torch.Tensor]:
    coordinates = torch.randn(residues, 14, 3, generator=torch.Generator().manual_seed(5))
    mask = torch.ones(residues, 14, dtype=torch.bool)
    return {
        "atom14_coordinates": coordinates,
        "model_atom_mask": mask,
        "coordinate_mask": mask,
        "resolved_atom_mask": mask,
        "aatype": torch.arange(residues) % 20,
        "res_idx": torch.arange(1, residues + 1),
        "chain_idx": torch.zeros(residues, dtype=torch.long),
        "chain_breaks_per_residue": torch.zeros(residues, dtype=torch.bool),
    }


def test_ca_corruption_exposes_and_supervises_only_ca() -> None:
    corrupted = corrupt_structure(
        _clean_structure(),
        sigma=1.0,
        generator=torch.Generator().manual_seed(7),
        translation_std=0.0,
        atom_representation="ca",
    )

    for key in ("model_atom_mask", "coordinate_mask", "resolved_atom_mask"):
        mask = corrupted[key]
        assert mask[..., 1].all()
        assert mask.sum().item() == len(mask)
    assert torch.count_nonzero(corrupted["x0"][..., 0, :]) == 0
    assert torch.count_nonzero(corrupted["x_t"][..., 2:, :]) == 0
    torch.testing.assert_close(
        corrupted["x0"][..., 1, :].mean(0),
        torch.zeros(3),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_ca_model_enforces_mask_even_for_atom14_caller() -> None:
    config = ModelConfig(
        node_dim=16,
        condition_dim=8,
        pair_dim=8,
        atom_dim=8,
        attention_heads=2,
        attention_head_dim=8,
        atom_heads=2,
        atom_head_dim=4,
        residue_encoder_depth=0,
        coarse_depth=1,
        residue_decoder_depth=0,
        atom_representation="ca",
    )
    model = HierarchicalKaveh(config)
    coordinates = torch.randn(1, 3, 14, 3)
    inputs = DenoiserInput(
        coordinates=coordinates,
        sigma=torch.ones(1, 3, 14),
        residue_index=torch.arange(3)[None],
        chain_index=torch.zeros(1, 3, dtype=torch.long),
        chain_break=torch.zeros(1, 3, dtype=torch.bool),
        atom_mask=torch.ones(1, 3, 14, dtype=torch.bool),
    )

    atom_mask, residue_mask = model._validate(inputs)
    assert residue_mask.all()
    assert atom_mask[..., 1].all()
    assert atom_mask.sum().item() == 3


class _CaDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = replace(ModelConfig(), atom_representation="ca")
        self.seen_masks: list[torch.Tensor] = []

    def forward(self, inputs, **_kwargs):
        self.seen_masks.append(inputs.atom_mask)
        batch, residues = inputs.coordinates.shape[:2]
        return Prediction(
            coordinates=inputs.coordinates * inputs.atom_mask[..., None],
            aatype_logits=torch.zeros(batch, residues, 20),
        )


def test_ca_sampling_keeps_non_ca_state_zero() -> None:
    model = _CaDenoiser()
    result = sample(
        model,
        (4,),
        batch_size=1,
        config=replace(SamplingConfig(), num_steps=2, gamma=0.0),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(9),
    )
    assert all(mask.sum().item() == 4 for mask in model.seen_masks)
    assert torch.count_nonzero(result.coordinates[..., 0, :]) == 0
    assert torch.count_nonzero(result.coordinates[..., 2:, :]) == 0
    assert result.topology.atom_mask[..., 1].all()


def test_ca_pdb_contains_one_atom_per_residue(tmp_path: Path) -> None:
    coordinates = torch.randn(1, 3, 14, 3)
    aatype = torch.tensor([[0, 7, 19]])
    mask = build_topology((3,), 1, "cpu", atom_representation="ca").atom_mask
    write_sample_batch(tmp_path, coordinates, aatype, (3,), atom_mask=mask)

    atom_lines = [
        line
        for line in (tmp_path / "sample_00000.pdb").read_text().splitlines()
        if line.startswith("ATOM")
    ]
    assert len(atom_lines) == 3
    assert all(line[12:16].strip() == "CA" for line in atom_lines)


def test_zero_lddt_weight_does_not_build_lddt_objective(monkeypatch) -> None:
    monkeypatch.setattr(
        losses_module,
        "smooth_lddt_loss",
        lambda *_args, **_kwargs: pytest.fail("zero-weight lDDT must not execute"),
    )
    coordinates = torch.randn(1, 3, 14, 3)
    mask = torch.zeros(1, 3, 14, dtype=torch.bool)
    mask[..., 1] = True
    prediction = Prediction(
        coordinates=coordinates.clone(),
        aatype_logits=torch.zeros(1, 3, 20),
    )
    inputs = DenoiserInput(
        coordinates=coordinates,
        sigma=torch.ones(1, 3, 14),
        residue_index=torch.arange(3)[None],
        chain_index=torch.zeros(1, 3, dtype=torch.long),
        chain_break=torch.zeros(1, 3, dtype=torch.bool),
        atom_mask=mask,
    )
    batch = {
        "x0": coordinates,
        "sigma": torch.ones(1),
        "coordinate_mask": mask,
        "resolved_atom_mask": mask,
        "residue_mask": mask[..., 1],
        "aatype": torch.zeros(1, 3, dtype=torch.long),
    }
    config = LossConfig(
        aatype_weight=0.25,
        smooth_lddt_weight=0.0,
        distogram_weight=0.0,
    )

    losses = losses_module.compute_losses(
        prediction,
        inputs,
        batch,
        config,
        ModelConfig(),
    )
    torch.testing.assert_close(losses["smooth_lddt_loss"], torch.zeros(()))
