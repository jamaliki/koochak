from __future__ import annotations

from pathlib import Path


def test_jobs_package_has_no_private_cluster_strings() -> None:
    root = Path(__file__).resolve().parents[1] / "koochak" / "jobs"
    haystack = "\n".join(path.read_text() for path in root.rglob("*.py"))
    forbidden = [
        "To" + "kyo",
        "sand" + "pit",
        "login." + "sand" + "pit",
        "/mnt/" + "lustre/users/",
    ]

    for token in forbidden:
        assert token not in haystack
