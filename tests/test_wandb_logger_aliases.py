import types
import sys

from koochak.logging.wandb_logger import make_wandb_hooks


class _FakeRun:
    def __init__(self):
        self.summary = {}
        self.id = "run123"
        self.name = "run-name"


class _FakeWandb:
    def __init__(self):
        self._run = _FakeRun()
        self._artifacts = []

    # API used by the logger
    def init(self, **kwargs):  # noqa: D401
        self._inited = kwargs
        return self._run

    @property
    def run(self):
        return self._run

    class Settings:
        def __init__(self, start_method=None):
            self.start_method = start_method

    class Artifact:
        def __init__(self, name, type, metadata=None):  # noqa: A002
            self.name = name
            self.type = type
            self.metadata = metadata or {}
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    def log(self, *args, **kwargs):
        pass

    def log_artifact(self, art, aliases=None):
        self._artifacts.append((art, list(aliases or [])))

    def finish(self):
        pass


def test_wandb_artifact_aliases(monkeypatch, tmp_path):
    # Install fake wandb module
    fake = _FakeWandb()
    sys.modules['wandb'] = fake  # type: ignore

    cfg = {
        "enabled": True,
        "project": "proj",
        "artifact_name_prefix": "model",
        "log_artifacts": True,
    }
    hooks = make_wandb_hooks(cfg)

    # Start training
    for fn in hooks.get("on_train_start", []):
        fn({"config_json": {}})

    # Simulate an eval where val_loss improves at step 10
    for fn in hooks.get("on_eval_end", []):
        fn({"val_loss": 0.5}, {"step": 10})

    # Simulate a checkpoint at step 10
    ckpt_path = str(tmp_path / "step000000010.pt")
    tmp_path.joinpath("step000000010.pt").write_text("x")
    for fn in hooks.get("on_checkpoint", []):
        fn(ckpt_path, {"step": 10}, {"step": 10})

    # Verify artifact was logged with expected aliases
    assert fake._artifacts, "no artifacts were logged"
    art, aliases = fake._artifacts[-1]
    assert art.name.startswith("model-")
    assert "latest" in aliases and "step-10" in aliases and "best" in aliases and "best-val_loss" in aliases


def test_wandb_artifacts_are_opt_in(monkeypatch, tmp_path):
    fake = _FakeWandb()
    sys.modules["wandb"] = fake  # type: ignore

    hooks = make_wandb_hooks({"enabled": True, "project": "proj"})

    for fn in hooks.get("on_train_start", []):
        fn({"config_json": {}})

    ckpt_path = str(tmp_path / "step000000010.pt")
    tmp_path.joinpath("step000000010.pt").write_text("x")
    for fn in hooks.get("on_checkpoint", []):
        fn(ckpt_path, {"step": 10}, {"step": 10})

    assert fake._artifacts == []
