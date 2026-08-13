from __future__ import annotations

from pathlib import Path


def test_repository_has_no_private_cluster_strings() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        *root.rglob("*.py"),
        *root.rglob("*.md"),
        *root.rglob("*.yaml"),
        *root.rglob("*.toml"),
    ]
    haystack = "\n".join(
        source.read_text()
        for source in sources
        if ".git" not in source.parts and "__pycache__" not in source.parts
    )
    forbidden = [
        "To" + "kyo",
        "sand" + "pit",
        "login." + "sand" + "pit",
        "/mnt/" + "lustre/users/",
        "kiarash-" + "eitgbi",
    ]

    for token in forbidden:
        assert token not in haystack
