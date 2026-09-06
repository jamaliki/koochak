from __future__ import annotations

import math

import pytest
import torch

from hierarchical_kaveh.config import LossConfig, ModelConfig
from hierarchical_kaveh.diffusion import (
    compute_losses,
    smooth_lddt_loss,
    smooth_lddt_sigma_weights,
)
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.model.attention import AtomToResidue, smooth_sigma_gate
from hierarchical_kaveh.types import DenoiserInput, Prediction
from tests.test_model import sample_input


def _small_config(**changes: object) -> ModelConfig:
    values: dict[str, object] = {
        "node_dim": 32,
        "condition_dim": 16,
        "pair_dim": 8,
        "atom_dim": 16,
        "attention_heads": 4,
        "attention_head_dim": 8,
        "atom_heads": 2,
        "atom_head_dim": 8,
        "atom_encoder_depth": 1,
        "residue_encoder_depth": 1,
        "coarse_depth": 2,
        "residue_decoder_depth": 1,
        "atom_decoder_depth": 1,
        "residue_ffn_expansion": 2,
        "atom_ffn_expansion": 2,
        "dropout": 0.0,
        "pair_rbf_bins": 4,
        "distogram_bins": 8,
    }
    values.update(changes)
    return ModelConfig(**values)


def _identity_pooler(**changes: object) -> AtomToResidue:
    pooler = AtomToResidue(2, 2, 1, **changes)
    with torch.no_grad():
        pooler.mean.weight.copy_(torch.eye(2))
        pooler.score.weight.zero_()
        pooler.output.weight.zero_()
    return pooler


def test_direct_intervention_defaults_preserve_legacy_configuration() -> None:
    model_config = ModelConfig()
    loss_config = LossConfig()
    assert model_config.atom_ffn_residual_scale == "full"
    assert model_config.atom_to_residue_mean_rmsnorm is False
    assert model_config.atom_to_residue_residual_scale == "full"
    assert model_config.atom_to_residue_transport == "all_atom"
    assert loss_config.smooth_lddt_sigma_max is None
    assert loss_config.smooth_lddt_resolved_atom_only is False
    assert loss_config.smooth_lddt_c_out_compensation is False


def test_lddt_sigma_weights_gate_and_compensate_with_explicit_c_out_formula() -> None:
    sigma = torch.tensor([0.0, 1.0, 4.0])
    gated = smooth_lddt_sigma_weights(sigma, sigma_max=3.0)
    torch.testing.assert_close(gated, torch.tensor([1.0, 1.0, 0.0]))

    compensated = smooth_lddt_sigma_weights(
        sigma, sigma_data=16.0, c_out_compensation=True
    )
    c_out = sigma * 16.0 / (sigma.square() + 16.0**2).sqrt()
    expected = 1.0 / c_out.clamp_min(torch.finfo(torch.float32).tiny)
    torch.testing.assert_close(compensated, expected)
    assert torch.isfinite(compensated).all()
    torch.testing.assert_close(
        compensated * c_out, torch.tensor([0.0, 1.0, 1.0])
    )


def test_lddt_sigma_gate_ignores_only_high_sigma_examples() -> None:
    torch.manual_seed(7)
    target = torch.randn(2, 3, 1, 3)
    prediction = target.clone()
    prediction[0, 0, 0, 0] += 1.0
    altered = prediction.clone()
    altered[1, 0, 0, 0] += 100.0
    atom_mask = torch.ones(2, 3, 1, dtype=torch.bool)
    sigma = torch.tensor([1.0, 4.0])

    ungated = smooth_lddt_loss(prediction, target, atom_mask, chunk_size=2)
    ungated_altered = smooth_lddt_loss(altered, target, atom_mask, chunk_size=2)
    gated = smooth_lddt_loss(
        prediction,
        target,
        atom_mask,
        chunk_size=2,
        sigma=sigma,
        sigma_max=3.0,
    )
    gated_altered = smooth_lddt_loss(
        altered,
        target,
        atom_mask,
        chunk_size=2,
        sigma=sigma,
        sigma_max=3.0,
    )

    assert not torch.allclose(ungated, ungated_altered)
    torch.testing.assert_close(gated, gated_altered)
    single_gated = smooth_lddt_loss(
        prediction[:1],
        target[:1],
        atom_mask[:1],
        chunk_size=2,
        sigma=sigma[:1],
        sigma_max=3.0,
    )
    torch.testing.assert_close(gated, single_gated)


def test_lddt_sigma_gate_with_no_active_examples_is_finite_zero() -> None:
    prediction = torch.randn(2, 3, 1, 3, requires_grad=True)
    target = torch.randn_like(prediction)
    atom_mask = torch.ones(2, 3, 1, dtype=torch.bool)
    loss = smooth_lddt_loss(
        prediction,
        target,
        atom_mask,
        sigma=torch.tensor([4.0, 5.0]),
        sigma_max=3.0,
    )
    torch.testing.assert_close(loss, torch.zeros(()))
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_resolved_lddt_mask_excludes_virtual_slots_but_coordinate_mse_keeps_them() -> None:
    target = torch.randn(1, 3, 14, 3)
    prediction = target.clone()
    prediction[..., 4:, :] += 10.0
    coordinate_mask = torch.ones(1, 3, 14, dtype=torch.bool)
    resolved_atom_mask = torch.zeros_like(coordinate_mask)
    resolved_atom_mask[..., :4] = True

    all_atom = smooth_lddt_loss(
        prediction, target, coordinate_mask, chunk_size=7
    )
    resolved = smooth_lddt_loss(
        prediction, target, resolved_atom_mask, chunk_size=7
    )
    physical_reference = smooth_lddt_loss(
        target, target, resolved_atom_mask, chunk_size=7
    )
    assert all_atom > resolved
    torch.testing.assert_close(resolved, physical_reference)

    batch_size, residues = 1, 3
    batch = {
        "x0": target,
        "sigma": torch.full((batch_size,), 0.25),
        "model_atom_mask": coordinate_mask,
        "coordinate_mask": coordinate_mask,
        "resolved_atom_mask": resolved_atom_mask,
        "residue_mask": coordinate_mask[..., 1],
        "aatype": torch.zeros(batch_size, residues, dtype=torch.long),
    }
    inputs = DenoiserInput(
        coordinates=prediction,
        sigma=torch.full_like(prediction[..., 0], 0.25),
        residue_index=torch.arange(residues)[None],
        chain_index=torch.zeros(batch_size, residues, dtype=torch.long),
        chain_break=torch.zeros(batch_size, residues, dtype=torch.bool),
        atom_mask=coordinate_mask,
    )
    base = LossConfig(
        align_coordinate_loss=False,
        aatype_weight=0.0,
        smooth_lddt_weight=1.0,
        distogram_weight=0.0,
    )
    physical_only = LossConfig(
        align_coordinate_loss=False,
        aatype_weight=0.0,
        smooth_lddt_weight=1.0,
        smooth_lddt_resolved_atom_only=True,
        distogram_weight=0.0,
    )
    prediction_object = Prediction(
        coordinates=prediction,
        aatype_logits=torch.zeros(batch_size, residues, 20),
    )
    base_losses = compute_losses(
        prediction_object, inputs, batch, base, ModelConfig()
    )
    physical_losses = compute_losses(
        prediction_object, inputs, batch, physical_only, ModelConfig()
    )
    assert base_losses["coordinate_loss"] > 0
    torch.testing.assert_close(
        physical_losses["coordinate_loss"], base_losses["coordinate_loss"]
    )
    assert physical_losses["smooth_lddt_loss"] < base_losses["smooth_lddt_loss"]


def test_lddt_c_out_compensation_cancels_raw_update_chain_factor_without_changing_prediction() -> None:
    target = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]]
    )
    delta = torch.tensor(
        [[[[0.1, 0.0, 0.0]], [[0.0, 0.1, 0.0]], [[0.0, 0.0, 0.1]]]]
    )
    atom_mask = torch.ones(1, 3, 1, dtype=torch.bool)
    sigma = torch.tensor([4.0])
    c_out = sigma * 16.0 / (sigma.square() + 16.0**2).sqrt()

    def gradient(compensate: bool) -> tuple[torch.Tensor, torch.Tensor]:
        raw = ((target + delta) / c_out[..., None, None, None]).detach()
        raw.requires_grad_()
        prediction = c_out[..., None, None, None] * raw
        loss = smooth_lddt_loss(
            prediction,
            target,
            atom_mask,
            sigma=sigma,
            c_out_compensation=compensate,
        )
        loss.backward()
        return raw.grad.detach(), prediction.detach()

    uncorrected_gradient, uncorrected_prediction = gradient(False)
    corrected_gradient, corrected_prediction = gradient(True)
    torch.testing.assert_close(corrected_prediction, uncorrected_prediction)
    assert torch.linalg.vector_norm(corrected_gradient) > 0
    torch.testing.assert_close(
        uncorrected_gradient,
        c_out * corrected_gradient,
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_lddt_c_out_compensation_is_finite_and_chain_matched_at_small_sigma() -> None:
    target = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]]
    )
    delta = torch.tensor(
        [[[[0.1, 0.0, 0.0]], [[0.0, 0.1, 0.0]], [[0.0, 0.0, 0.1]]]]
    )
    atom_mask = torch.ones(1, 3, 1, dtype=torch.bool)
    sigma = torch.tensor([1.0e-4])
    c_out = sigma * 16.0 / (sigma.square() + 16.0**2).sqrt()
    weight = smooth_lddt_sigma_weights(
        sigma, sigma_data=16.0, c_out_compensation=True
    )
    assert weight.item() > 1_000.0
    assert torch.isfinite(weight).all()

    raw = ((target + delta) / c_out[..., None, None, None]).detach().requires_grad_()
    prediction = c_out[..., None, None, None] * raw
    compensated_loss = smooth_lddt_loss(
        prediction,
        target,
        atom_mask,
        sigma=sigma,
        c_out_compensation=True,
    )
    compensated_loss.backward()

    direct_prediction = prediction.detach().requires_grad_()
    direct_loss = smooth_lddt_loss(direct_prediction, target, atom_mask)
    direct_loss.backward()

    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert direct_prediction.grad is not None
    torch.testing.assert_close(raw.grad, direct_prediction.grad, atol=1.0e-5, rtol=1.0e-4)


def test_backbone_first_transport_has_exact_all_atom_and_backbone_only_endpoints() -> None:
    pooler = _identity_pooler(transport="backbone_first")
    atoms = torch.zeros(1, 1, 14, 2)
    atoms[..., :4, 0] = 1.0
    atoms[..., 4:, 1] = 2.0
    condition = torch.zeros(1, 1, 1)
    mask = torch.ones(1, 1, 14, dtype=torch.bool)

    low_noise = pooler(atoms, condition, mask, sigma=torch.tensor([1.0]))
    high_noise = pooler(atoms, condition, mask, sigma=torch.tensor([6.0]))
    torch.testing.assert_close(low_noise, torch.tensor([[[4.0 / 14.0, 20.0 / 14.0]]]))
    torch.testing.assert_close(high_noise, torch.tensor([[[1.0, 0.0]]]))

    sidechain_perturbed = atoms.clone()
    sidechain_perturbed[..., 4:, 1] = 7.0
    torch.testing.assert_close(
        pooler(sidechain_perturbed, condition, mask, sigma=torch.tensor([6.0])),
        high_noise,
    )
    assert not torch.allclose(
        pooler(sidechain_perturbed, condition, mask, sigma=torch.tensor([1.0])),
        low_noise,
    )

    torch.testing.assert_close(
        smooth_sigma_gate(torch.tensor([1.0, 2.0, 5.0, 6.0]), full_sigma=2.0, zero_sigma=5.0),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )


def test_atom_mean_rmsnorm_changes_only_the_pooled_mean_projection() -> None:
    pooler = _identity_pooler(mean_rmsnorm=True)
    atoms = torch.zeros(1, 1, 14, 2)
    atoms[..., 0, :] = torch.tensor([1.0, 2.0])
    atoms[..., 1, :] = torch.tensor([2.0, 2.0])
    mask = torch.zeros(1, 1, 14, dtype=torch.bool)
    mask[..., :2] = True
    output = pooler(atoms, torch.zeros(1, 1, 1), mask)
    pooled = torch.tensor([[[1.5, 2.0]]])
    expected = pooler.mean_norm(pooled)
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, pooled)


def test_atom_ffn_and_atom_to_residue_depth_scales_are_independently_wired() -> None:
    full_model = HierarchicalKaveh(
        _small_config(
            atom_ffn_residual_scale="full",
            atom_to_residue_residual_scale="full",
        )
    )
    depth_model = HierarchicalKaveh(
        _small_config(
            atom_ffn_residual_scale="depth",
            atom_to_residue_residual_scale="depth",
        )
    )
    expected_depth_scale = 1.0 / math.sqrt(2.0 * (1 + 2 + 1))
    for block in (*full_model.atom_encoder, *full_model.atom_decoder):
        assert float(block.ffn.residual_scale) == pytest.approx(1.0)
    for block in (*depth_model.atom_encoder, *depth_model.atom_decoder):
        assert float(block.ffn.residual_scale) == pytest.approx(expected_depth_scale)
    assert float(full_model.atom_to_residue.residual_scale) == pytest.approx(1.0)
    assert float(depth_model.atom_to_residue.residual_scale) == pytest.approx(
        expected_depth_scale
    )


def test_model_forward_accepts_the_full_atom14_intervention_bundle() -> None:
    model = HierarchicalKaveh(
        _small_config(
            atom_ffn_residual_scale="depth",
            atom_to_residue_mean_rmsnorm=True,
            atom_to_residue_residual_scale="depth",
            atom_to_residue_transport="backbone_first",
            attention_residual_scale="depth",
            sandwich_rmsnorm=True,
        )
    ).eval()
    output = model(sample_input(lengths=(5,), padded=5), compute_distogram=False)
    assert torch.isfinite(output.coordinates).all()
    assert torch.isfinite(output.aatype_logits).all()
