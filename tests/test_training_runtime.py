from __future__ import annotations

from unittest.mock import Mock

import torch
from torch import nn

from hierarchical_kaveh.config import LossConfig, ModelConfig
from hierarchical_kaveh.training import PallatomTrainingStep, denoiser_input
from hierarchical_kaveh.types import Prediction


def _batch() -> dict[str, torch.Tensor]:
    batch, residues = 2, 4
    atom_mask = torch.ones(batch, residues, 14, dtype=torch.bool)
    return {
        "x_t": torch.randn(batch, residues, 14, 3),
        "x0": torch.randn(batch, residues, 14, 3),
        "t": torch.full((batch, residues, 14), 0.1),
        "sigma": torch.full((batch,), 0.1),
        "model_atom_mask": atom_mask,
        "coordinate_mask": atom_mask,
        "residue_mask": atom_mask[..., 1],
        "aatype": torch.randint(0, 20, (batch, residues)),
        "aatype_input": torch.full((batch, residues), 20),
        "res_idx": torch.arange(residues).expand(batch, -1),
        "chain_idx": torch.zeros(batch, residues, dtype=torch.long),
        "chain_breaks_per_residue": torch.zeros(batch, residues, dtype=torch.bool),
    }


def _prediction(batch: dict[str, torch.Tensor]) -> Prediction:
    batch_size, residues = batch["aatype"].shape
    return Prediction(
        coordinates=batch["x_t"] * 0.5,
        aatype_logits=torch.randn(batch_size, residues, 20),
        distogram=torch.randn(batch_size, residues, residues, 64),
    )


class FakeModel(nn.Module):
    def __init__(self, prediction: Prediction):
        super().__init__()
        self.prediction = prediction
        self.inputs = []

    def forward(self, inputs, **kwargs):
        self.inputs.append((inputs, kwargs))
        return self.prediction


def test_denoiser_input_uses_standard_edm_contract() -> None:
    batch = _batch()
    inputs = denoiser_input(batch)
    assert inputs.coordinates is batch["x_t"]
    assert inputs.sigma is batch["t"]
    assert inputs.self_conditioned_coordinates is None


def test_training_step_always_self_conditions_coordinates(monkeypatch) -> None:
    batch = _batch()
    prediction = _prediction(batch)
    model = FakeModel(prediction)
    loss = prediction.coordinates.sum() * 0 + 3.0
    compute_losses = Mock(
        return_value={
            "loss": loss,
            "coordinate_loss": loss + 1,
            "aatype_loss": loss + 2,
            "aatype_active_fraction": loss + 2.5,
            "smooth_lddt_loss": loss + 3,
            "distogram_loss": loss + 4,
        }
    )
    monkeypatch.setattr("hierarchical_kaveh.training.compute_losses", compute_losses)
    step = PallatomTrainingStep(LossConfig(), ModelConfig())
    output = step(model, batch, {"autocast": torch.no_grad})

    assert len(model.inputs) == 2
    first_input, first_kwargs = model.inputs[0]
    final_input, final_kwargs = model.inputs[1]
    assert torch.equal(final_input.self_conditioned_coordinates, prediction.coordinates)
    assert not final_input.self_conditioned_coordinates.requires_grad
    assert first_kwargs == {"compute_distogram": False}
    assert final_kwargs == {"compute_distogram": True}
    compute_losses.assert_called_once_with(
        prediction,
        final_input,
        batch,
        step.loss_config,
        step.model_config,
        patch_distogram_implementation="auto",
    )
    assert output["loss"] is loss
    assert output["self_conditioned"].item() == 1.0


def test_training_step_can_skip_self_conditioning_and_distogram(monkeypatch) -> None:
    batch = _batch()
    batch["data_owned_shard_count"] = torch.tensor(17)
    batch["koochak_prefetch_get_wait_s"] = 0.125
    prediction = _prediction(batch)
    model = FakeModel(prediction)
    loss = prediction.coordinates.sum() * 0 + 3.0
    compute_losses = Mock(
        return_value={
            "loss": loss,
            "coordinate_loss": loss + 1,
            "aatype_loss": loss + 2,
            "aatype_active_fraction": loss + 2.5,
            "smooth_lddt_loss": loss + 3,
            "distogram_loss": loss + 4,
        }
    )
    monkeypatch.setattr("hierarchical_kaveh.training.compute_losses", compute_losses)
    step = PallatomTrainingStep(
        LossConfig(distogram_weight=0.0),
        ModelConfig(),
        self_conditioning_probability=0.0,
    )
    output = step(model, batch, {"autocast": torch.no_grad, "step": 17})

    assert len(model.inputs) == 1
    final_input, final_kwargs = model.inputs[0]
    assert final_input.self_conditioned_coordinates is None
    assert final_kwargs == {"compute_distogram": False}
    assert output["self_conditioned"].item() == 0.0
    assert output["data_owned_shard_count"].item() == 17
    assert output["koochak_prefetch_get_wait_s"] == 0.125


def test_half_self_conditioning_schedule_is_deterministic_and_nontrivial() -> None:
    left = PallatomTrainingStep(
        LossConfig(), ModelConfig(), self_conditioning_probability=0.5, seed=42
    )
    right = PallatomTrainingStep(
        LossConfig(), ModelConfig(), self_conditioning_probability=0.5, seed=42
    )
    left_schedule = [left._use_self_conditioning(step) for step in range(64)]
    right_schedule = [right._use_self_conditioning(step) for step in range(64)]
    assert left_schedule == right_schedule
    assert any(left_schedule)
    assert not all(left_schedule)
