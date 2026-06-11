from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch
import torch.distributed as dist

from ..utils import flags


@flags.compile_wrap
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@flags.compile_wrap
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim > 2:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # Use Moonlight update
    update *= (0.2 * max(grad.size(-2), grad.size(-1)) ** 0.5)
    return update


@flags.compile_wrap
def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps).float()
    ################ NorMuon added ###################
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))  # Keeps the update norm the same as pre-normalization
    ##################################################
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


@flags.compile_wrap
def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# ---------------------------------------------------------------------------
# Shared step helpers (private)
# ---------------------------------------------------------------------------

_ParamUpdate = Callable[[torch.nn.Parameter, Dict[str, Any], Dict[str, Any]], torch.Tensor]


def _optimizer_profile_ranges_enabled() -> bool:
    return os.environ.get("KAVEH_PROFILE_RANGES", "").lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _optimizer_profile_range(name: str):
    if not _optimizer_profile_ranges_enabled():
        yield
        return
    with torch.autograd.profiler.record_function(name):
        yield


def _foreach_adam_enabled(group: Dict[str, Any]) -> bool:
    if "foreach_adam_update" in group:
        return bool(group["foreach_adam_update"])
    return os.environ.get("KAVEH_ADAM_FOREACH_UPDATE", "").lower() in {"1", "true", "yes", "on"}


def _ensure_grad(p: torch.nn.Parameter) -> None:
    """Make sure `p.grad` is materialized so collective ops don't desync ranks."""
    if p.grad is None:
        p.grad = torch.zeros_like(p)


def _apply_weight_decay_and_step(p: torch.nn.Parameter, update: torch.Tensor, group: Dict[str, Any]) -> None:
    p.mul_(1 - group["lr"] * group["weight_decay"])
    p.add_(update.reshape(p.shape), alpha=-group["lr"])


def _muon_update_for_param(p: torch.nn.Parameter, group: Dict[str, Any], state: Dict[str, Any]) -> torch.Tensor:
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(p)
    return muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])


def _normuon_update_for_param(p: torch.nn.Parameter, group: Dict[str, Any], state: Dict[str, Any]) -> torch.Tensor:
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(p)
        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
    return normuon_update(
        p.grad,
        state["momentum_buffer"],
        state["second_momentum_buffer"],
        beta=group["momentum"],
        beta2=group["beta2"],
    )


def _adam_step_for_param(p: torch.nn.Parameter, group: Dict[str, Any], state: Dict[str, Any]) -> torch.Tensor:
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] = 0
    state["step"] += 1
    return adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"])


def _single_device_muon_group(
    optimizer: torch.optim.Optimizer,
    group: Dict[str, Any],
    update_fn: _ParamUpdate,
) -> None:
    for p in group["params"]:
        _ensure_grad(p)
        update = update_fn(p, group, optimizer.state[p])
        _apply_weight_decay_and_step(p, update, group)


def _distributed_muon_group(
    optimizer: torch.optim.Optimizer,
    group: Dict[str, Any],
    update_fn: _ParamUpdate,
) -> None:
    params: List[torch.nn.Parameter] = group["params"]
    if not params:
        return
    world = dist.get_world_size()
    rank = dist.get_rank()
    # Match upstream Muon: always extend by `world - n%world`, even when n%world == 0.
    params_pad: List[torch.nn.Parameter] = (
        params + [torch.empty_like(params[-1])] * (world - len(params) % world)
    )
    for base_i in range(0, len(params), world):
        local_index = base_i + rank
        if local_index < len(params):
            p = params[local_index]
            _ensure_grad(p)
            update = update_fn(p, group, optimizer.state[p])
            _apply_weight_decay_and_step(p, update, group)
        dist.all_gather(params_pad[base_i:base_i + world], params_pad[base_i + rank])


def _single_device_adam_group(optimizer: torch.optim.Optimizer, group: Dict[str, Any]) -> None:
    if _foreach_adam_enabled(group) and _foreach_adam_group(optimizer, group):
        return
    for p in group["params"]:
        _ensure_grad(p)
        update = _adam_step_for_param(p, group, optimizer.state[p])
        _apply_weight_decay_and_step(p, update, group)


def _foreach_adam_group(optimizer: torch.optim.Optimizer, group: Dict[str, Any]) -> bool:
    params = list(group["params"])
    if not params:
        return True
    devices = {p.device for p in params}
    dtypes = {p.dtype for p in params}
    if len(devices) != 1 or len(dtypes) != 1:
        return False
    grads: list[torch.Tensor] = []
    exp_avgs: list[torch.Tensor] = []
    exp_avg_sqs: list[torch.Tensor] = []
    states: list[Dict[str, Any]] = []
    old_steps: list[int] = []
    for p in params:
        _ensure_grad(p)
        state = optimizer.state[p]
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] = 0
        states.append(state)
        old_steps.append(int(state["step"]))
    if len(set(old_steps)) != 1:
        return False
    step = old_steps[0] + 1
    for p, state in zip(params, states):
        state["step"] = step
        grads.append(p.grad)  # type: ignore[arg-type]
        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

    beta1, beta2 = group["betas"]
    sqrt_bias_correction2 = (1 - beta2 ** step) ** 0.5
    step_size = group["lr"] * sqrt_bias_correction2 / (1 - beta1 ** step)

    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    grad_squares = torch._foreach_mul(grads, grads)
    torch._foreach_lerp_(exp_avg_sqs, grad_squares, 1 - beta2)
    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_add_(denoms, group["eps"] * sqrt_bias_correction2)
    weight_decay = float(group["weight_decay"])
    if weight_decay != 0.0:
        torch._foreach_mul_(params, 1 - group["lr"] * weight_decay)
    torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-step_size)
    return True


def _sorted_by_size_desc(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return sorted(list(params), key=lambda x: x.size(), reverse=True)


def _apply_muon_group_defaults(group: Dict[str, Any], *, with_beta2: bool = False) -> None:
    group.setdefault("lr", 0.02)
    group.setdefault("momentum", 0.95)
    group.setdefault("weight_decay", 0)
    if with_beta2:
        group.setdefault("beta2", 0.95)


def _apply_adam_group_defaults(group: Dict[str, Any]) -> None:
    group.setdefault("lr", 3e-4)
    group.setdefault("betas", (0.9, 0.95))
    group.setdefault("eps", 1e-10)
    group.setdefault("weight_decay", 0)


def _assert_muon_param_list(params: Sequence[Any]) -> None:
    assert (
        isinstance(params, list)
        and len(params) >= 1
        and isinstance(params[0], torch.nn.Parameter)
    )


def _maybe_apply_closure(closure):
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()


# ---------------------------------------------------------------------------
# Single-group Muon variants
# ---------------------------------------------------------------------------


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        _assert_muon_param_list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(_sorted_by_size_desc(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            _distributed_muon_group(self, group, _muon_update_for_param)
        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon variant for usage in non-distributed settings."""

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            _single_device_muon_group(self, group, _muon_update_for_param)
        return loss


class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        _assert_muon_param_list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(_sorted_by_size_desc(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            _distributed_muon_group(self, group, _normuon_update_for_param)
        return loss


class SingleDeviceNorMuon(torch.optim.Optimizer):
    """Muon variant for usage in non-distributed settings."""

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            _single_device_muon_group(self, group, _normuon_update_for_param)
        return loss


# ---------------------------------------------------------------------------
# WithAuxAdam variants
# ---------------------------------------------------------------------------


def _init_aux_adam_groups(
    param_groups: Sequence[Dict[str, Any]],
    *,
    with_beta2: bool,
    sort_muon: bool,
) -> None:
    for group in param_groups:
        assert "use_muon" in group
        if group["use_muon"]:
            if sort_muon:
                group["params"] = _sorted_by_size_desc(group["params"])
            _apply_muon_group_defaults(group, with_beta2=with_beta2)
        else:
            _apply_adam_group_defaults(group)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        _init_aux_adam_groups(param_groups, with_beta2=False, sort_muon=True)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            if group["use_muon"]:
                with _optimizer_profile_range("optimizer.muon.update"):
                    _distributed_muon_group(self, group, _muon_update_for_param)
            else:
                with _optimizer_profile_range("optimizer.adam.update"):
                    _single_device_adam_group(self, group)
        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed variant of MuonWithAuxAdam."""

    def __init__(self, param_groups):
        _init_aux_adam_groups(param_groups, with_beta2=False, sort_muon=False)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            if group["use_muon"]:
                with _optimizer_profile_range("optimizer.muon.update"):
                    _single_device_muon_group(self, group, _muon_update_for_param)
            else:
                with _optimizer_profile_range("optimizer.adam.update"):
                    _single_device_adam_group(self, group)
        return loss


class NorMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        _init_aux_adam_groups(param_groups, with_beta2=True, sort_muon=True)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            if group["use_muon"]:
                with _optimizer_profile_range("optimizer.muon.update"):
                    _distributed_muon_group(self, group, _normuon_update_for_param)
            else:
                with _optimizer_profile_range("optimizer.adam.update"):
                    _single_device_adam_group(self, group)
        return loss


class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        _init_aux_adam_groups(param_groups, with_beta2=True, sort_muon=False)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = _maybe_apply_closure(closure)
        for group in self.param_groups:
            if group["use_muon"]:
                with _optimizer_profile_range("optimizer.muon.update"):
                    _single_device_muon_group(self, group, _normuon_update_for_param)
            else:
                with _optimizer_profile_range("optimizer.adam.update"):
                    _single_device_adam_group(self, group)
        return loss
