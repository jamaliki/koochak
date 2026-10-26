"""Observe completed updates on every rank before their metrics reach loggers."""

import multiprocessing as mp
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from koochak.loop import training_loop


def run_case(folder, hooks, *, steps=3, checkpoint=None, ema=True, log_every=1):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    saved = training_loop(model=model, optimizer=optimizer, scheduler=scheduler,
        dataset=[torch.ones(1, 1)] * steps,
        step_fn=lambda module, batch, context: dict(loss=module(batch).square().mean()),
        checkpoint_dict=checkpoint, hooks=hooks,
        train_cfg=dict(ddp=False, device='cpu', max_steps=steps, log_every=log_every,
            grad_clip_norm=None, out_dir=str(folder), ckpt_every=100,
            ema=dict(enabled=ema, decay=.5, offload_to_cpu=False)))
    return saved, model, optimizer


def test_update_observation_precedes_log_and_step_end_with_same_log(tmp_path):
    events, logs, identities = [], [], []

    def update(log, context):
        step = context['step']
        events.append(('update', step))
        identities.append(id(log))
        assert context['optimizer'].param_groups[0]['lr'] == pytest.approx(.1 * .5 ** (step + 1))
        assert context['ema'].num_updates == step + 1
        log['stability/completed_updates'] = step + 1

    def logged(log, context):
        events.append(('log', context['step']))
        assert id(log) == identities[-1]
        logs.append(dict(log))

    def ended(log, context):
        events.append(('end', context['step']))
        assert id(log) == identities[-1]
        assert log['stability/completed_updates'] == context['step'] + 1

    _, _, optimizer = run_case(tmp_path, dict(on_update_end=[update], on_log=[logged], on_step_end=[ended]))
    assert events == [(kind, step) for step in range(3) for kind in ('update', 'log', 'end')]
    assert [log['stability/completed_updates'] for log in logs] == [1, 2, 3]
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.0125)


def test_update_context_keeps_current_optimizer_and_resumed_ema(tmp_path):
    first, _, _ = run_case(tmp_path / 'first', {}, steps=1)
    observed = []

    def update(log, context):
        ema, optimizer = context['ema'], context['optimizer']
        assert ema.num_updates == 2
        assert optimizer.state
        observed.append((ema, optimizer))
        current = context['model'].weight
        expected = .5 * first['ema']['shadow']['weight'] + .5 * current
        torch.testing.assert_close(ema.shadow['weight'], expected)

    resumed, _, optimizer = run_case(tmp_path / 'resumed', dict(on_update_end=[update]),
                                    steps=2, checkpoint=first)
    assert len(observed) == 1 and observed[0][1] is optimizer
    assert resumed['ema']['num_updates'] == 2
    torch.testing.assert_close(observed[0][0].shadow['weight'], resumed['ema']['shadow']['weight'])


def test_update_hook_runs_without_ema_and_between_log_updates(tmp_path):
    updates, logs = [], []

    def update(log, context):
        assert context['ema'] is None
        assert isinstance(context['optimizer'], torch.optim.Adam)
        assert 'loss' in log and 'step_time_s' in log
        updates.append(context['step'])
        log['observed'] = context['step']

    run_case(tmp_path, dict(on_update_end=[update], on_log=[lambda row, ctx: logs.append(dict(row))]),
             ema=False, log_every=2)
    assert updates == [0, 1, 2]
    assert [log['observed'] for log in logs] == [0, 2]


@pytest.mark.parametrize('fail_closed', [False, True])
def test_update_errors_follow_existing_hook_policy(tmp_path, fail_closed):
    logs = []

    def reject(log, context):
        raise ValueError('Invalid update observation')

    reject._koochak_fail_closed = fail_closed
    hooks = dict(on_update_end=[reject], on_log=[lambda row, ctx: logs.append(ctx['step'])])
    if fail_closed:
        with pytest.raises(ValueError, match='Invalid update observation'):
            run_case(tmp_path, hooks)
        assert logs == []
    else:
        run_case(tmp_path, hooks)
        assert logs == [0, 1, 2]


def test_observation_cost_stays_outside_existing_step_timer(tmp_path, monkeypatch):
    import koochak.loop as loop
    clock, measured = [0.], []
    monkeypatch.setattr(loop.time, 'perf_counter', lambda: clock[0])

    def update(log, context):
        clock[0] += 10

    run_case(tmp_path, dict(on_update_end=[update],
        on_log=[lambda row, ctx: measured.append(row['step_time_s'])]))
    assert measured == [0., 0., 0.]


def distributed_worker(rank, init_file, folder):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=f'file://{init_file}',
        rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        updates, logs = [], []

        def update(log, context):
            assert context['optimizer'] is optimizer
            assert (context['ema'] is None) == (rank == 1)
            total = torch.tensor(float(rank + 1))
            torch.distributed.all_reduce(total)
            assert total.item() == 3
            log['rank_sum'] = total.item()
            updates.append(context['step'])

        def logged(log, context):
            assert log['rank_sum'] == 3
            logs.append(context['step'])

        training_loop(model=model, optimizer=optimizer, dataset=[torch.ones(1, 1)] * 2,
            step_fn=lambda module, batch, context: dict(loss=module(batch).square().mean()),
            train_cfg=dict(ddp=True, device='cpu', warn_unsharded=False, max_steps=2,
                log_every=1, out_dir=folder, ckpt_every=100, grad_clip_norm=None,
                ema=dict(enabled=True, decay=.5, offload_to_cpu=False)),
            hooks=dict(on_update_end=[update], on_log=[logged]))
        torch.save(dict(updates=updates, logs=logs), Path(folder) / f'observed-{rank}.pt')
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason='Gloo is unavailable')
def test_update_hook_runs_on_every_rank_before_rank_zero_logging(tmp_path):
    context = mp.get_context('spawn')
    folder = tmp_path / 'run'
    folder.mkdir()
    processes = [context.Process(target=distributed_worker,
        args=(rank, str(tmp_path / 'init'), str(folder))) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    for rank in range(2):
        observed = torch.load(folder / f'observed-{rank}.pt', weights_only=True)
        assert observed == dict(updates=[0, 1], logs=[0, 1] if rank == 0 else [])
