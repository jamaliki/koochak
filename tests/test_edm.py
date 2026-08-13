import math

import torch

from hierarchical_kaveh.diffusion import sample_training_sigma, sigma_from_probability


def test_scaled_lognormal_training_schedule_is_reproducible() -> None:
    first = torch.Generator().manual_seed(123)
    second = torch.Generator().manual_seed(123)
    values = torch.stack([sample_training_sigma(first) for _ in range(2048)])
    copies = torch.stack([sample_training_sigma(second) for _ in range(2048)])
    torch.testing.assert_close(values, copies)
    log_scaled = (values / 16.0).log()
    assert abs(float(log_scaled.mean()) - (-1.2)) < 0.08
    assert abs(float(log_scaled.std()) - 1.5) < 0.08


def test_probability_schedule_matches_lognormal_quantiles() -> None:
    probabilities = torch.tensor([0.1, 0.5, 0.9])
    sigmas = sigma_from_probability(probabilities)
    assert torch.all(sigmas[1:] > sigmas[:-1])
    assert math.isclose(float(sigmas[1]), 16.0 * math.exp(-1.2), rel_tol=1e-6)
