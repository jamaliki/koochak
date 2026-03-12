import hashlib
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
        self.watch_calls = []
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self._run

    @property
    def run(self):
        return self._run

    class Settings:
        def __init__(self, start_method=None):
            self.start_method = start_method

    def watch(self, model, **kwargs):
        self.watch_calls.append((model, kwargs))

    def finish(self):
        pass


def test_wandb_named_run_uses_stable_id_and_disables_watch_by_default():
    fake = _FakeWandb()
    sys.modules["wandb"] = fake  # type: ignore

    hooks = make_wandb_hooks(
        {
            "enabled": True,
            "project": "proj",
            "name": "compile-run",
        }
    )

    for fn in hooks.get("on_train_start", []):
        fn({"config_json": {}, "train_cfg": {"compile": {"enabled": True}}, "model": object()})

    assert fake.init_kwargs is not None
    assert fake.watch_calls == []
    assert fake.init_kwargs["resume"] is None
    assert fake.init_kwargs["id"] == hashlib.sha1(b"compile-run").hexdigest()


def test_wandb_resume_allow_uses_stable_id():
    fake = _FakeWandb()
    sys.modules["wandb"] = fake  # type: ignore

    hooks = make_wandb_hooks(
        {
            "enabled": True,
            "project": "proj",
            "name": "resume-name",
            "resume": "allow",
        }
    )

    for fn in hooks.get("on_train_start", []):
        fn({"config_json": {}, "train_cfg": {"compile": {"enabled": False}}, "model": object()})

    expected = hashlib.sha1(b"resume-name").hexdigest()
    assert fake.init_kwargs is not None
    assert fake.init_kwargs["resume"] == "allow"
    assert fake.init_kwargs["id"] == expected


def test_wandb_explicit_id_overrides_name_hash():
    fake = _FakeWandb()
    sys.modules["wandb"] = fake  # type: ignore

    hooks = make_wandb_hooks(
        {
            "enabled": True,
            "project": "proj",
            "name": "resume-name",
            "id": "manual-id",
            "resume": "allow",
        }
    )

    for fn in hooks.get("on_train_start", []):
        fn({"config_json": {}, "train_cfg": {"compile": {"enabled": False}}, "model": object()})

    assert fake.init_kwargs is not None
    assert fake.init_kwargs["id"] == "manual-id"
