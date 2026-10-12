import torch

from koochak.loop import training_loop
from koochak.storage import checkpoint


def train(folder, steps, saved=None):
    model = torch.nn.Linear(1, 1)
    evaluations = []
    result = training_loop(model=model, dataset=[torch.ones(1, 1)] * steps,
        step_fn=lambda module, batch, context: {"loss": module(batch).square().mean()},
        optimizer=torch.optim.SGD(model.parameters(), lr=.1), checkpoint_dict=saved,
        eval_dataset=[torch.ones(1, 1)],
        eval_fn=lambda module, data, context: evaluations.append(context["step"] + 1) or {},
        train_cfg=dict(ddp=False, max_steps=steps, ckpt_every=2, eval_every=2,
                       intervals_use_completed_steps=True, out_dir=str(folder), keep_last_k=0))
    return result, evaluations


def test_completed_intervals_survive_resume_at_checkpoint_boundary(tmp_path):
    result, evaluated = train(tmp_path, 2)
    assert evaluated == [2]
    resumed, evaluated = train(tmp_path, 5, result)
    assert evaluated == [4]
    assert resumed["next_step"] == 5
    assert sorted(file.name for file in tmp_path.glob("step*.pt")) == [
        "step000000002.pt", "step000000004.pt", "step000000005.pt"]
    saved = checkpoint.load(str(tmp_path / "step000000004.pt"))
    assert saved["step"] == saved["next_step"] == 4
