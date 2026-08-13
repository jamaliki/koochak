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
