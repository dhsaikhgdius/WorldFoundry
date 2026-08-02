"""Stochastic flow transition math shared by rollout and exact replay."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, pi


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("stochastic flow transitions require the 'train-core' extra") from error
    return torch


def flow_sigma_for_sample(value: object, sample: object, *, field_name: str) -> object:
    torch = _require_torch()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=sample.device, dtype=torch.float32)
    else:
        value = value.to(device=sample.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(int(sample.shape[0]))
    elif value.numel() == int(sample.shape[0]):
        value = value.reshape(int(sample.shape[0]))
    else:
        raise ValueError(f"{field_name} must be scalar or contain one value per sample")
    return value.reshape((int(sample.shape[0]),) + (1,) * (sample.ndim - 1))


@dataclass(frozen=True, slots=True)
class FlowSDETransition:
    next_sample: object
    mean: object
    scale: object
    log_prob: object | None


def flow_match_sigma_schedule(num_steps: int, *, shift: float = 3.0) -> tuple[float, ...]:
    """Return the shifted static FlowMatch sigma schedule.

    The base grid is ``linspace(1, 0, T + 1)`` in FP32 and the time shift is
    ``shift*t / (1 + (shift-1)*t)``.  Returning the actual FP32 values keeps
    recipe serialization and runtime tensors on the same numerical contract.
    """

    torch = _require_torch()
    if isinstance(num_steps, bool) or int(num_steps) <= 0:
        raise ValueError("num_steps must be a positive integer")
    resolved_shift = float(shift)
    if not isfinite(resolved_shift) or resolved_shift <= 0:
        raise ValueError("shift must be finite and positive")
    base = torch.linspace(1.0, 0.0, int(num_steps) + 1, dtype=torch.float32)
    sigmas = resolved_shift * base / (1.0 + (resolved_shift - 1.0) * base)
    return tuple(float(value) for value in sigmas.tolist())


def gaussian_transition_log_prob(sample: object, mean: object, scale: object) -> object:
    """Mean element log-likelihood for each sample in a diagonal Gaussian."""

    torch = _require_torch()
    if not all(torch.is_tensor(value) for value in (sample, mean, scale)):
        raise TypeError("Gaussian log-prob inputs must be torch.Tensor values")
    if sample.shape != mean.shape or sample.ndim < 2:
        raise ValueError("sample and mean must share shape [B,...]")
    try:
        scale = torch.broadcast_to(scale, sample.shape)
    except RuntimeError as error:
        raise ValueError("scale cannot broadcast to sample") from error
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("Gaussian scale must be finite and positive")
    elementwise = (
        -(sample.detach().float() - mean.float()).square() / (2.0 * scale.float().square())
        - torch.log(scale.float())
        - 0.5 * log(2.0 * pi)
    )
    return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))


def flow_sde_transition(
    velocity: object,
    sample: object,
    sigma: object,
    sigma_next: object,
    *,
    eta: float,
    sigma_max: float = 0.99,
    generator: object | None = None,
    next_sample: object | None = None,
    trajectory_dtype: object | None = None,
) -> FlowSDETransition:
    """Sample or replay one Flow-GRPO transition in explicit FP32 math.

    Supplying ``next_sample`` switches the function to replay: the exact stored
    latent is scored under the newly computed transition mean.  The latent is
    round-tripped through ``trajectory_dtype`` before log-prob evaluation so
    rollout and replay agree when trajectories are stored in BF16/FP16.
    """

    torch = _require_torch()
    if not torch.is_tensor(velocity) or not torch.is_tensor(sample):
        raise TypeError("velocity and sample must be torch.Tensor values")
    if velocity.shape != sample.shape or sample.ndim < 2:
        raise ValueError("velocity and sample must share shape [B,...]")
    resolved_eta = float(eta)
    resolved_sigma_max = float(sigma_max)
    if not isfinite(resolved_eta) or resolved_eta < 0:
        raise ValueError("eta must be finite and non-negative")
    if not isfinite(resolved_sigma_max) or not 0 < resolved_sigma_max < 1:
        raise ValueError("sigma_max must be finite and in (0,1)")
    if trajectory_dtype is None:
        trajectory_dtype = sample.dtype
    if trajectory_dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("trajectory_dtype must be a floating torch dtype")

    current = sample.float()
    prediction = velocity.float()
    sigma_value = flow_sigma_for_sample(sigma, current, field_name="sigma")
    next_sigma_value = flow_sigma_for_sample(sigma_next, current, field_name="sigma_next")
    if not bool(torch.isfinite(sigma_value).all() and torch.isfinite(next_sigma_value).all()):
        raise ValueError("sigmas must be finite")
    if not bool((sigma_value > 0).all() and (sigma_value <= 1).all()):
        raise ValueError("sigma must be in (0,1]")
    if not bool((next_sigma_value >= 0).all() and (next_sigma_value < sigma_value).all()):
        raise ValueError("sigma_next must be in [0,sigma)")

    dt = next_sigma_value - sigma_value
    denominator_sigma = torch.where(sigma_value == 1, resolved_sigma_max, sigma_value)
    diffusion = torch.sqrt(sigma_value / (1.0 - denominator_sigma)) * resolved_eta
    mean = (
        current * (1.0 + diffusion.square() / (2.0 * sigma_value) * dt)
        + prediction * (1.0 + diffusion.square() * (1.0 - sigma_value) / (2.0 * sigma_value)) * dt
    )
    scale = diffusion * torch.sqrt(-dt)

    if next_sample is None:
        if resolved_eta < 1.0e-7:
            sampled = mean
        else:
            noise = torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=generator,
            )
            sampled = mean + scale * noise
    else:
        if not torch.is_tensor(next_sample) or next_sample.shape != sample.shape:
            raise ValueError("next_sample must match sample shape")
        sampled = next_sample.float()

    stored = sampled.to(dtype=trajectory_dtype)
    if resolved_eta < 1.0e-7:
        log_prob = None
    else:
        scored = stored.float()
        log_prob = gaussian_transition_log_prob(scored, mean, scale)
    return FlowSDETransition(next_sample=stored, mean=mean, scale=scale, log_prob=log_prob)


def flow_ode_step(velocity: object, sample: object, sigma: object, sigma_next: object) -> object:
    """Deterministic Euler step used outside the stochastic training window."""

    torch = _require_torch()
    if not torch.is_tensor(velocity) or not torch.is_tensor(sample) or velocity.shape != sample.shape:
        raise ValueError("velocity and sample must be matching torch.Tensor values")
    current = sample.float()
    dt = flow_sigma_for_sample(sigma_next, current, field_name="sigma_next") - flow_sigma_for_sample(
        sigma,
        current,
        field_name="sigma",
    )
    if not bool((dt < 0).all()):
        raise ValueError("ODE schedule must be strictly descending")
    return current + velocity.float() * dt


__all__ = [
    "FlowSDETransition",
    "flow_match_sigma_schedule",
    "flow_ode_step",
    "flow_sigma_for_sample",
    "flow_sde_transition",
    "gaussian_transition_log_prob",
]
