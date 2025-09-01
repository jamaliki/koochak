from koochak.storage.naming import parse_step_from_name, parse_step_from_path, make_checkpoint_aliases


def test_parse_step_from_name_and_path():
    assert parse_step_from_name("step000000123.pt") == 123
    assert parse_step_from_path("/tmp/step000000045.pt") == 45
    assert parse_step_from_name("checkpoint.pt") is None


def test_make_checkpoint_aliases_basic():
    aliases = make_checkpoint_aliases(10)
    assert "latest" in aliases
    assert "step-10" in aliases
    assert "best" not in aliases


def test_make_checkpoint_aliases_with_best():
    aliases = make_checkpoint_aliases(5, best_keys=["val_loss", "acc"])  # type: ignore[arg-type]
    assert set(["latest", "step-5", "best", "best-val_loss", "best-acc"]).issubset(set(aliases))

