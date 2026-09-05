import pytest
import torch

from scripts.mixture_data_canary import _validate_progres_batch


def test_validate_progres_batch_accepts_normalized_inputs() -> None:
    embedding = torch.zeros(3, 128)
    embedding[:, 0] = 1.0
    result = _validate_progres_batch({"progres_embedding": embedding})
    assert result["progres_batch_size"] == 3
    assert result["progres_embedding_dim"] == 128
    assert result["progres_norm_error_max"] == 0.0


@pytest.mark.parametrize("embedding", [torch.zeros(2, 127), torch.full((2, 128), float("nan"))])
def test_validate_progres_batch_rejects_invalid_inputs(embedding: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="Progres"):
        _validate_progres_batch({"progres_embedding": embedding})
