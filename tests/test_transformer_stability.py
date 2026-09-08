from __future__ import annotations

import math

import pytest
import torch

from hierarchical_kaveh.config import ModelConfig
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.model.attention import _normalize_qkv
from hierarchical_kaveh.model.layers import (
    DiTAdaLNZero,
    NonAffineRMSNorm,
    UnitRMSNorm,
    apply_residual_stage,
)
from tests.test_model import sample_input, small_config


def test_attention_residual_scale_is_strictly_validated() -> None:
    with pytest.raises(ValueError, match="attention_residual_scale"):
        ModelConfig(attention_residual_scale="invalid")


def test_block_conditioning_style_is_strictly_validated() -> None:
    with pytest.raises(ValueError, match="block_conditioning_style"):
        ModelConfig(block_conditioning_style="almost_dit")


def test_gain_invariant_modes_are_strictly_validated() -> None:
    with pytest.raises(ValueError, match="qk_norm_mode"):
        ModelConfig(qk_norm_mode="learned_temperature")
    with pytest.raises(ValueError, match="pair_residual_mode"):
        ModelConfig(pair_residual_mode="tanh_scalar")


def test_dit_adaln_zero_uses_the_canonical_modulation_equation() -> None:
    module = DiTAdaLNZero(width=4, condition_dim=3)
    assert len(tuple(module.parameters())) == 2
    x = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])
    condition = torch.randn(1, 1, 3)
    shift = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    scale = torch.tensor([[[0.5, -0.25, 0.0, 1.0]]])

    actual = module.attention_input(x, shift, scale)
    expected = module.norm1(x) * (1.0 + scale) + shift

    torch.testing.assert_close(actual, expected)
    assert all(
        torch.count_nonzero(value) == 0
        for value in module.modulation_parameters(condition)
    )


def test_dit_adaln_zero_starts_as_identity_and_opens_through_gates() -> None:
    model = HierarchicalKaveh(
        small_config(block_conditioning_style="dit_adaln_zero")
    )
    block = model.residue_encoder[0]
    assert block.adaln_zero is not None
    assert block.attention.gate is None
    assert torch.count_nonzero(block.attention.output.weight) > 0
    assert torch.count_nonzero(block.ffn.down.weight) > 0

    x = torch.randn(2, 5, model.config.node_dim, requires_grad=True)
    condition = torch.randn(2, 5, model.config.condition_dim)
    mask = torch.ones(2, 5, dtype=torch.bool)
    positions = torch.arange(5)[None].expand(2, -1)
    output, diagnostics = block(x, condition, mask, positions)

    torch.testing.assert_close(output, x)
    torch.testing.assert_close(diagnostics[:2, 0], torch.zeros(2))
    output.sum().backward()

    projection = block.adaln_zero.adaLN_modulation[-1]
    assert projection.weight.grad is not None
    gradient_chunks = projection.weight.grad.chunk(6, dim=0)
    assert torch.count_nonzero(gradient_chunks[2]) > 0
    assert torch.count_nonzero(gradient_chunks[5]) > 0
    assert block.attention.output.weight.grad is not None
    assert torch.count_nonzero(block.attention.output.weight.grad) == 0
    assert block.ffn.down.weight.grad is not None
    assert torch.count_nonzero(block.ffn.down.weight.grad) == 0


def test_bounded_dit_removes_residual_gates_and_bounds_branch_modulation() -> None:
    module = DiTAdaLNZero(width=4, condition_dim=3, bounded=True)
    assert len(tuple(module.parameters())) == 2
    condition = torch.randn(2, 5, 3) * 100.0
    values = module.modulation_parameters(condition)
    assert len(values) == 4
    assert all(float(value.detach().abs().max()) <= 0.5 + 1e-6 for value in values)
    assert module.condition_norm.elementwise_affine is False
    assert isinstance(module.norm1, torch.nn.LayerNorm)
    assert all(torch.count_nonzero(value) == 0 for value in values)


def test_bounded_dit_full_model_forward_has_no_residual_gate_dependency() -> None:
    model = HierarchicalKaveh(
        small_config(
            block_conditioning_style="dit_bounded",
            qk_norm_mode="per_head_rms",
            pair_residual_mode="fixed_unit_rms",
        )
    ).eval()
    output = model(sample_input(), compute_distogram=False)
    assert torch.isfinite(output.coordinates).all()
    assert torch.isfinite(output.aatype_logits).all()


def test_fixed_gain_modes_remove_qk_affine_and_pair_scale_parameters() -> None:
    model = HierarchicalKaveh(
        small_config(
            attention_residual_scale="depth",
            sandwich_rmsnorm=True,
            block_conditioning_style="dit_bounded",
            qk_norm_mode="per_head_rms",
            pair_residual_mode="fixed_unit_rms",
        )
    )
    assert not any(
        name.endswith(suffix)
        for name, _ in model.named_parameters()
        for suffix in ("q_norm.weight", "q_norm.bias", "k_norm.weight", "k_norm.bias")
    )
    assert all(
        not block.pair_multiplication.outgoing_scale.requires_grad
        and not block.pair_multiplication.incoming_scale.requires_grad
        and isinstance(block.pair_multiplication.pair_post_norm, UnitRMSNorm)
        for block in model.coarse
    )


def test_per_head_qk_rms_is_parameter_free_and_unit_scale() -> None:
    qkv = torch.randn(2, 7, 3 * 4 * 8, dtype=torch.bfloat16)
    query, key, value = _normalize_qkv(
        qkv,
        None,
        None,
        heads=4,
        head_dim=8,
        mode="per_head_rms",
    )
    assert query.dtype == qkv.dtype
    assert key.dtype == qkv.dtype
    assert torch.equal(value, qkv[..., 2 * 4 * 8 :])
    for tensor in (query, key):
        head = tensor.float().reshape(2, 7, 4, 8)
        torch.testing.assert_close(
            head.square().mean(-1), torch.ones(2, 7, 4), atol=2e-2, rtol=2e-2
        )


def test_dit_adaln_zero_is_wired_to_every_node_transformer_block() -> None:
    model = HierarchicalKaveh(
        small_config(
            block_conditioning_style="dit_adaln_zero",
            coarse_pair_transition=True,
        )
    )
    residual_scale = 1.0 / math.sqrt(2.0 * (1 + 2 + 1))
    blocks = (
        *model.atom_encoder,
        *model.residue_encoder,
        *model.coarse,
        *model.residue_decoder,
        *model.atom_decoder,
    )
    for block in blocks:
        assert block.adaln_zero is not None
        assert float(block.attention_residual_scale) == pytest.approx(1.0)
    for block in (*model.residue_encoder, *model.coarse, *model.residue_decoder):
        assert float(block.ffn.residual_scale) == pytest.approx(residual_scale)
    for block in (*model.atom_encoder, *model.atom_decoder):
        assert float(block.ffn.residual_scale) == pytest.approx(1.0)
    for block in model.coarse:
        assert block.pair_ffn is not None
        assert float(block.pair_ffn.residual_scale) == pytest.approx(residual_scale)


def test_sandwich_norm_is_non_affine_and_normalizes_the_stream_after_add() -> None:
    norm = NonAffineRMSNorm()
    assert tuple(norm.parameters()) == ()

    stream = torch.full((2, 3, 8), 0.25)
    normalized = norm(stream)
    torch.testing.assert_close(
        normalized.float().square().mean(-1),
        torch.ones(2, 3),
        atol=1e-5,
        rtol=1e-5,
    )

    output, diagnostics = apply_residual_stage(
        torch.zeros_like(stream),
        stream,
        scale=torch.tensor(0.5),
        post_norm=norm,
        mask=torch.ones(2, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(
        output.float().square().mean(-1), torch.ones(2, 3), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        diagnostics[:3], torch.tensor([0.25, 0.125, 1.0]), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("scale_mode", ("full", "depth"))
@pytest.mark.parametrize("sandwich", (False, True))
def test_all_transformer_blocks_receive_the_selected_stability_factors(
    scale_mode: str, sandwich: bool
) -> None:
    model = HierarchicalKaveh(
        small_config(
            attention_residual_scale=scale_mode,
            sandwich_rmsnorm=sandwich,
            coarse_pair_transition=True,
        )
    )
    expected_attention_scale = (
        1.0
        if scale_mode == "full"
        else 1.0 / math.sqrt(2.0 * (1 + 2 + 1))
    )
    for block in (*model.atom_encoder, *model.atom_decoder):
        assert float(block.attention_residual_scale) == pytest.approx(expected_attention_scale)
        assert (block.attention_post_norm is not None) is sandwich
        assert (block.ffn_post_norm is not None) is sandwich
        # Atom FFNs are not part of the trunk-depth scaling contract.
        assert float(block.ffn.residual_scale) == pytest.approx(1.0)
    for block in (*model.residue_encoder, *model.residue_decoder):
        assert float(block.attention_residual_scale) == pytest.approx(expected_attention_scale)
        assert (block.attention_post_norm is not None) is sandwich
        assert (block.ffn_post_norm is not None) is sandwich
        assert float(block.ffn.residual_scale) == pytest.approx(
            1.0 / math.sqrt(2.0 * (1 + 2 + 1))
        )
    for block in model.coarse:
        assert float(block.attention_residual_scale) == pytest.approx(expected_attention_scale)
        assert (block.attention_post_norm is not None) is sandwich
        assert (block.ffn_post_norm is not None) is sandwich
        assert (block.pair_ffn_post_norm is not None) is sandwich
        assert float(block.ffn.residual_scale) == pytest.approx(
            1.0 / math.sqrt(2.0 * (1 + 2 + 1))
        )


@pytest.mark.parametrize("sandwich", (False, True))
def test_forward_exposes_finite_residual_diagnostics(sandwich: bool) -> None:
    model = HierarchicalKaveh(
        small_config(sandwich_rmsnorm=sandwich, coarse_pair_transition=True)
    ).eval()
    output = model(sample_input())
    assert output.residual_diagnostics is not None
    assert output.residual_diagnostics.shape == (3, 3)
    assert torch.isfinite(output.residual_diagnostics).all()
    if sandwich:
        torch.testing.assert_close(
            output.residual_diagnostics[:, 2],
            torch.ones(3),
            atol=1e-4,
            rtol=1e-4,
        )
    else:
        torch.testing.assert_close(
            output.residual_diagnostics[:, 2],
            output.residual_diagnostics[:, 1],
            atol=1e-7,
            rtol=1e-7,
        )
