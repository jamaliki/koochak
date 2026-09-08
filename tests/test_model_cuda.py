import pytest
import torch

from hierarchical_kaveh.model.attention import atom_attention_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_atom_attention_matches_reference():
    from hierarchical_kaveh.model.kernels.atom_window import atom_window_attention, supported

    torch.manual_seed(7)
    device = torch.device("cuda")
    shape = (2, 8, 14, 4, 32)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    atom_mask = torch.ones(shape[:3], device=device, dtype=torch.bool)
    atom_mask[1, 6:] = False
    segment = torch.tensor([[0] * 4 + [1] * 4, [0] * 8], device=device)
    if not supported(query, key, value, atom_mask, segment, 1):
        pytest.skip("Triton atom kernel does not support this GPU")
    expected = atom_attention_reference(query, key, value, atom_mask, segment, 1)
    actual = atom_window_attention(query, key, value, atom_mask, segment, 1, 32**-0.5)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_qk_layernorm_matches_native_forward_and_backward():
    from hierarchical_kaveh.model.kernels.qk_norm import normalize

    torch.manual_seed(11)
    device = torch.device("cuda")
    heads, head_dim = 12, 64
    qkv = torch.randn(4, 1024, 3 * heads * head_dim, device=device, dtype=torch.bfloat16)
    qkv_fused = qkv.detach().clone().requires_grad_()
    qkv_native = qkv.detach().clone().requires_grad_()
    q_norm_fused = torch.nn.LayerNorm(heads * head_dim, device=device).to(torch.bfloat16)
    k_norm_fused = torch.nn.LayerNorm(heads * head_dim, device=device).to(torch.bfloat16)
    q_norm_native = torch.nn.LayerNorm(heads * head_dim, device=device).to(torch.bfloat16)
    k_norm_native = torch.nn.LayerNorm(heads * head_dim, device=device).to(torch.bfloat16)
    q_norm_native.load_state_dict(q_norm_fused.state_dict())
    k_norm_native.load_state_dict(k_norm_fused.state_dict())

    fused = normalize(qkv_fused, q_norm_fused, k_norm_fused)
    q, k, value = qkv_native.chunk(3, dim=-1)
    native = (q_norm_native(q), k_norm_native(k), value)
    for actual, expected in zip(fused, native):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    fused_loss = sum(value.float().square().mean() for value in fused)
    native_loss = sum(value.float().square().mean() for value in native)
    fused_loss.backward()
    native_loss.backward()
    torch.testing.assert_close(qkv_fused.grad, qkv_native.grad, atol=5e-2, rtol=5e-2)
    for actual, expected in (
        (q_norm_fused.weight.grad, q_norm_native.weight.grad),
        (q_norm_fused.bias.grad, q_norm_native.bias.grad),
        (k_norm_fused.weight.grad, k_norm_native.weight.grad),
        (k_norm_fused.bias.grad, k_norm_native.bias.grad),
    ):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
