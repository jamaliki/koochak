import gc
from dataclasses import replace

import pytest
import torch

from hierarchical_kaveh.config import ModelConfig
from hierarchical_kaveh.model import HierarchicalKaveh, expand_distogram
from hierarchical_kaveh.model.attention import atom_window_mask
from hierarchical_kaveh.model.pair import PairMultiplicationBlock
from hierarchical_kaveh.types import CompactDistogram, DenoiserInput


def small_config(**changes) -> ModelConfig:
    values = dict(
        node_dim=32,
        condition_dim=16,
        pair_dim=8,
        atom_dim=16,
        attention_heads=4,
        attention_head_dim=8,
        atom_heads=2,
        atom_head_dim=8,
        atom_encoder_depth=1,
        residue_encoder_depth=1,
        coarse_depth=2,
        residue_decoder_depth=1,
        atom_decoder_depth=1,
        residue_ffn_expansion=2,
        atom_ffn_expansion=2,
        dropout=0.0,
        pair_rbf_bins=4,
        distogram_bins=8,
    )
    values.update(changes)
    return ModelConfig(**values)


def sample_input(lengths=(7, 5), padded=9) -> DenoiserInput:
    batch = len(lengths)
    positions = torch.arange(padded)[None].expand(batch, -1)
    residue_mask = positions < torch.tensor(lengths)[:, None]
    atom_mask = residue_mask[..., None].expand(-1, -1, 14).clone()
    coordinates = torch.randn(batch, padded, 14, 3)
    sigma = 0.1 + torch.rand(batch, padded, 14)
    chain_index = torch.zeros(batch, padded, dtype=torch.long)
    chain_break = torch.zeros(batch, padded, dtype=torch.long)
    if lengths[0] >= 4:
        chain_index[0, 3:lengths[0]] = 1
        chain_break[0, 3] = 1
    return DenoiserInput(
        coordinates=coordinates,
        sigma=sigma,
        residue_index=positions.clone(),
        chain_index=chain_index,
        chain_break=chain_break,
        atom_mask=atom_mask,
    )


def test_forward_contract_and_compact_distogram():
    model = HierarchicalKaveh(small_config()).eval()
    inputs = sample_input()
    prediction = model(inputs)

    assert prediction.coordinates.shape == (2, 9, 14, 3)
    assert prediction.aatype_logits.shape == (2, 9, 20)
    assert isinstance(prediction.distogram, CompactDistogram)
    assert expand_distogram(prediction.distogram).shape == (2, 9, 9, 8)
    assert torch.count_nonzero(prediction.coordinates[1, 5:]) == 0


def test_fixed_patch_capacity_matches_compact_eager_model_output():
    model = HierarchicalKaveh(small_config()).eval()
    inputs = sample_input()
    exact = model(inputs)
    static = model(replace(inputs, patch_capacity=4))

    torch.testing.assert_close(static.coordinates, exact.coordinates, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(static.aatype_logits, exact.aatype_logits, atol=1e-6, rtol=1e-6)
    assert static.distogram is not None and exact.distogram is not None
    torch.testing.assert_close(
        expand_distogram(static.distogram),
        expand_distogram(exact.distogram),
        atol=1e-6,
        rtol=1e-6,
    )


def test_local_center_geometry_and_intermediate_feedback_variant():
    model = HierarchicalKaveh(
        small_config(
            pair_geometry_mode="local_center",
            pair_self_conditioned_geometry=True,
            intermediate_distograms=True,
            intermediate_distogram_feedback=True,
        )
    ).eval()
    inputs = sample_input()
    previous = model(inputs, compute_distogram=False)
    output = model(
        inputs.with_self_conditioning(previous),
        compute_intermediate_distograms=True,
    )

    assert len(output.intermediate_distograms) == 1
    assert output.intermediate_distograms[0].coarse_logits.shape[:3] == (2, 2, 2)
    assert torch.isfinite(output.intermediate_distograms[0].coarse_logits).all()


def test_zero_initialized_coordinate_head_is_exact_edm_skip():
    model = HierarchicalKaveh(small_config()).eval()
    inputs = sample_input(lengths=(5,), padded=5)
    output = model(inputs, compute_distogram=False)
    denominator = inputs.sigma.square() + model.config.sigma_data**2
    expected = model.config.sigma_data**2 / denominator
    expected = expected[..., None] * inputs.coordinates
    assert torch.allclose(output.coordinates, expected, atol=1e-6, rtol=1e-6)
    assert output.distogram is None


def test_output_contract_is_finite():
    model = HierarchicalKaveh(small_config()).eval()
    output = model(sample_input(lengths=(5,), padded=5))
    assert torch.isfinite(output.coordinates).all()


def test_secondary_structure_conditioning_prediction_and_recycling() -> None:
    model = HierarchicalKaveh(
        small_config(
            secondary_structure_conditioning=True,
            secondary_structure_prediction=True,
            secondary_structure_self_conditioning=True,
        )
    ).eval()
    inputs = sample_input(lengths=(5,), padded=5)
    inputs = DenoiserInput(
        coordinates=inputs.coordinates,
        sigma=inputs.sigma,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
        chain_break=inputs.chain_break,
        atom_mask=inputs.atom_mask,
        aatype_input=inputs.aatype_input,
        secondary_structure_input=torch.tensor([[0, 1, 2, 3, 0]]),
    )
    first = model(inputs, compute_distogram=False)
    second = model(inputs.with_self_conditioning(first), compute_distogram=False)
    assert first.secondary_structure_logits is not None
    assert first.secondary_structure_logits.shape == (1, 5, 3)
    assert torch.isfinite(second.secondary_structure_logits).all()


def test_pair_multiplication_is_present_every_coarse_layer_with_separate_trainable_scales():
    model = HierarchicalKaveh(small_config(coarse_depth=3))
    blocks = [block.pair_multiplication for block in model.coarse]
    assert len(blocks) == 3
    assert all(isinstance(block, PairMultiplicationBlock) for block in blocks)
    assert all(isinstance(block.outgoing_scale, torch.nn.Parameter) for block in blocks)
    assert all(isinstance(block.incoming_scale, torch.nn.Parameter) for block in blocks)
    assert all(block.outgoing_scale is not block.incoming_scale for block in blocks)
    forbidden = ("triangle_attention", "node_to_pair", "geometry_refresh", "secondary_structure")
    assert not any(any(name in module_name for name in forbidden) for module_name, _ in model.named_modules())


def test_backward_reaches_atom_residue_and_pair_streams():
    model = HierarchicalKaveh(small_config())
    output = model(sample_input(lengths=(5,), padded=5))
    dense = expand_distogram(output.distogram)
    loss = (
        output.coordinates.square().mean()
        + output.aatype_logits.square().mean()
        + dense.square().mean()
    )
    loss.backward()
    expected = (
        "atom_input.coordinates.weight",
        "residue_encoder.0.attention.qkv.weight",
        "coarse.0.attention.qkv.weight",
        "coarse.0.pair_multiplication.outgoing.input_projection.weight",
        "residue_decoder.0.attention.qkv.weight",
        "atom_decoder.0.attention.qkv.weight",
        "distogram.slot_bias",
    )
    parameters = dict(model.named_parameters())
    assert all(parameters[name].grad is not None for name in expected)


def test_atom_window_mask_stops_at_chain_segments():
    atom_mask = torch.ones(1, 3, 14, dtype=torch.bool)
    segment = torch.tensor([[0, 0, 1]])
    mask = atom_window_mask(atom_mask, segment, radius=1)
    # Query residue 1 has keys from residues 0 and 1, never residue 2.
    assert mask[0, 1, :, :28].all()
    assert not mask[0, 1, :, 28:].any()


def test_default_configuration_is_the_promoted_h6_architecture():
    config = ModelConfig()
    assert (config.atom_encoder_depth, config.residue_encoder_depth) == (1, 4)
    assert (config.coarse_depth, config.residue_decoder_depth, config.atom_decoder_depth) == (6, 4, 1)
    assert (config.atom_dim, config.node_dim, config.condition_dim, config.pair_dim) == (128, 768, 256, 64)


def test_default_model_has_outgoing_and_incoming_multiplication_in_all_six_coarse_blocks():
    model = HierarchicalKaveh(ModelConfig())
    assert len(model.coarse) == 6
    for coarse in model.coarse:
        multiplication = coarse.pair_multiplication
        assert multiplication.outgoing.direction == "outgoing"
        assert multiplication.incoming.direction == "incoming"
        assert multiplication.outgoing_scale.requires_grad
        assert multiplication.incoming_scale.requires_grad
        assert multiplication.outgoing_scale is not multiplication.incoming_scale
    assert not any("triangle_attention" in name for name, _ in model.named_modules())
    del model
    gc.collect()


def test_strict_fused_backend_flag_is_fail_loud(monkeypatch):
    from hierarchical_kaveh.model.backend import fused_failure

    monkeypatch.setenv("HIERARCHICAL_KAVEH_REQUIRE_FUSED", "1")
    with pytest.raises(RuntimeError, match="required fused backend unavailable"):
        fused_failure("test operator")
