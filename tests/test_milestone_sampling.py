from __future__ import annotations

import pytest

from scripts.sample_short128_milestone import _parse_lengths


def test_milestone_lengths_are_positive_and_unique() -> None:
    assert _parse_lengths("64,96,128") == (64, 96, 128)
    with pytest.raises(ValueError, match="positive"):
        _parse_lengths("64,0")
    with pytest.raises(ValueError, match="unique"):
        _parse_lengths("64,64")
