import pytest

from scripts import submit_transformer_stability_factorial as launcher


def test_progres_weights_are_pinned_to_verified_code_root() -> None:
    expected = launcher.REMOTE_CODE_ROOT / "progres-data" / "v1.1.0"
    assert launcher.PROGRES_DATA == expected
    assert launcher.progres_data_spec()["root"] == str(expected)
    assert launcher.progres_data_spec()["files"]["trained_model.pt"]["path"] == str(
        expected / "trained_model.pt"
    )
    assert launcher.progres_data_spec()["files"]["cath40.pt"]["path"] == str(
        expected / "cath40.pt"
    )


def test_progres_weights_reject_faulty_run_root() -> None:
    faulty = launcher.REMOTE_RUN_ROOT / "progres-data" / "v1.1.0"
    with pytest.raises(ValueError, match="Progres data must use"):
        launcher.progres_data_spec(faulty)


def test_progres_attestation_is_a_dependency_of_every_analysis() -> None:
    source = (launcher.REPO_ROOT / "scripts/submit_transformer_stability_factorial.py").read_text()
    assert '"attest-progres-data"' in source
    assert 'str(PROGRES_DATA)' in source
    assert '"progres-data/attestation.json"' in source


def test_attestation_is_not_the_faulty_run_root() -> None:
    spec = launcher.progres_data_spec()
    assert spec["root"] != str(launcher.REMOTE_RUN_ROOT / "progres-data" / "v1.1.0")
    assert {entry["md5"] for entry in spec["files"].values()} == {
        "c490293eb8d0bb350e68a8229c6884da",
        "616510a3b21d45500b26dd5ae9a040d6",
    }
