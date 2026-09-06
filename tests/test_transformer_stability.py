from __future__ import annotations

import math

import pytest
import torch

from hierarchical_kaveh.config import ModelConfig
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.model.layers import NonAffineRMSNorm, apply_residual_stage
from tests.test_model import sample_input, small_config


def test_attention_residual_scale_is_strictly_validated() -> None:
    with pytest.raises(ValueError, match="attention_residual_scale"):
        ModelConfig(attention_residual_scale="invalid")


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
