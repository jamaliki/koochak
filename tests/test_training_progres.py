from dataclasses import dataclass

from scripts.analyze_training_progres import _choose, _topology


@dataclass(frozen=True)
class Reference:
    index: int


def test_training_sample_selection_is_deterministic_and_without_replacement() -> None:
    references = [Reference(index) for index in range(100)]
    first = _choose(references, 16, 7)
    second = _choose(references, 16, 7)
    assert first == second
    assert len(set(first)) == 16
    assert first != _choose(references, 16, 8)


def test_cath_topology_uses_first_three_levels() -> None:
    assert _topology("1.25.40.10 - Tetratricopeptide repeat domain") == (
        "1.25.40",
        "1.25.40.10",
    )
