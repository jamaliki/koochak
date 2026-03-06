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
        self.logged = []

    def init(self, **kwargs):  # noqa: D401
        self._inited = kwargs
        return self._run

    @property
    def run(self):
        return self._run

    class Settings:
        def __init__(self, start_method=None):
            self.start_method = start_method

    def log(self, payload, **kwargs):
        self.logged.append((dict(payload), dict(kwargs)))

    def finish(self):
        pass


def test_wandb_hooks_forward_metric_keys_unchanged():
    fake = _FakeWandb()
    sys.modules["wandb"] = fake  # type: ignore

    hooks = make_wandb_hooks(
        {
            "enabled": True,
            "project": "proj",
            "name": "metric-forwarding-test",
        }
    )

    for fn in hooks.get("on_train_start", []):
        fn({"config_json": {}})

    train_metrics = {
        "loss": 1.25,
        "iou_semantic_non_empty": 0.8,
        "iou_semantic_protein_backbone": 0.5,
    }
    for fn in hooks.get("on_log", []):
        fn(train_metrics, {"step": 7, "train_cfg": {"eval_every": 0}})

    eval_metrics = {
        "val_loss": 0.75,
        "val_iou_semantic_non_empty": 0.85,
        "val_iou_semantic_protein_backbone": 0.6,
    }
    for fn in hooks.get("on_eval_end", []):
        fn(eval_metrics, {"step": 7})

    assert fake.logged[0][0]["iou_semantic_non_empty"] == 0.8
    assert fake.logged[0][0]["iou_semantic_protein_backbone"] == 0.5
    assert fake.logged[0][1]["step"] == 7
    assert fake.logged[1][0]["val_iou_semantic_non_empty"] == 0.85
    assert fake.logged[1][0]["val_iou_semantic_protein_backbone"] == 0.6
    assert fake.logged[1][1]["step"] == 7
