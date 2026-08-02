"""Exact FlowMap and on-policy DMD formulas used by AnyFlow.

Key formulas:
  - FlowMap interpolation: x_t = t * eps + (1 - t) * x_0
  - FlowMap Euler step: x_r = x_t - (t - r) * u(x_t, t, r)
  - FlowMap target: u* = (eps - x_0) - (t - r) * dF/dt
  - Central difference: dF/dt ≈ (F(t+ε) - F(t-ε)) / (2ε * guidance)
  - DMD gradient: g = (x0_fake - x0_real) / mean|x_gen - x0_real|
  - DMD proxy: L = ||x_gen - stop(x_gen - g)||^2 (fp64 MSE injects grad g)
  - On-policy real guidance: x_real = x_cond + (x_cond - x_uncond) * w

References:
  - AnyFlow (Flow Map + on-policy DMD): https://arxiv.org/abs/2605.13724
  - DMD: https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

import torch
from torch import Tensor


def _time_coefficients(timesteps: Tensor, reference: Tensor) -> Tensor:
    if not isinstance(reference, Tensor) or reference.ndim != 5:
        raise TypeError("AnyFlow latent reference must be BCTHW")
    if not isinstance(timesteps, Tensor):
        timesteps = torch.as_tensor(timesteps, device=reference.device, dtype=torch.float32)
    value = timesteps.to(device=reference.device, dtype=torch.float32)
    batch, frames = int(reference.shape[0]), int(reference.shape[2])
    if value.ndim == 0:
        value = value.expand(batch)
    if value.ndim == 1 and value.shape == (batch,):
        return value.reshape(batch, 1, 1, 1, 1)
    if value.ndim == 2 and value.shape == (batch, frames):
        return value.reshape(batch, 1, frames, 1, 1)
    raise ValueError("timesteps must be scalar, [B], or [B,T]")


def shift_flowmap_time(timesteps: Tensor, shift: float) -> Tensor:
    """Apply ``shift*t / (1 + (shift-1)*t)`` in normalized time."""

    value = torch.as_tensor(timesteps)
    scale = float(shift)
    if scale <= 0:
        raise ValueError("FlowMap timestep shift must be positive")
    return scale * value / (1.0 + (scale - 1.0) * value)


def flowmap_interpolate(clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
    """Return ``x_t = t*noise + (1-t)*clean`` for BCTHW latents."""

    if clean.shape != noise.shape:
        raise ValueError("clean and noise tensors must have identical shapes")
    time = _time_coefficients(timesteps, clean).to(dtype=clean.dtype)
    return time * noise + (1.0 - time) * clean


def flowmap_step(
    sample: Tensor,
    velocity: Tensor,
    timesteps: Tensor,
    destination_timesteps: Tensor,
) -> Tensor:
    """Euler FlowMap update ``x_r = x_t - (t-r) u(x_t,t,r)``."""

    if sample.shape != velocity.shape:
        raise ValueError("FlowMap sample and velocity must have identical shapes")
    current = _time_coefficients(timesteps, sample).to(dtype=sample.dtype)
    destination = _time_coefficients(destination_timesteps, sample).to(dtype=sample.dtype)
    return sample - (current - destination) * velocity


def flowmap_inference_schedule(
    step_count: int,
    *,
    shift: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return shifted ``[1,...,0]`` interval boundaries for any step count."""

    if isinstance(step_count, bool) or int(step_count) <= 0:
        raise ValueError("step_count must be a positive integer")
    base = torch.linspace(1.0, 0.0, int(step_count) + 1, device=device, dtype=dtype)
    return shift_flowmap_time(base, shift)


def beta08_train_weight(
    timesteps: Tensor,
    *,
    num_train_timesteps: int,
    shift: float,
) -> Tensor:
    """Nearest-grid beta08 weight from the official FlowMap scheduler."""

    if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) <= 1:
        raise ValueError("num_train_timesteps must be an integer greater than one")
    count = int(num_train_timesteps)
    device = timesteps.device
    grid = torch.linspace(1.0, 0.0, count + 1, device=device, dtype=torch.float64)[:-1]
    grid = shift_flowmap_time(grid, shift)
    raw = grid * torch.sqrt((1.0 - grid).clamp_min(0.0))
    weights = raw * (count / raw.sum())
    values = timesteps.to(device=device, dtype=torch.float64) / count
    indices = torch.argmin(
        (grid.reshape(-1, 1) - values.reshape(1, -1)).abs(),
        dim=0,
    )
    return weights[indices].reshape(timesteps.shape).to(dtype=torch.float32)


def allocate_flowmap_intervals(
    first: Tensor,
    second: Tensor,
    *,
    global_start_index: int,
    global_batch_size: int,
    diffusion_ratio: float,
    consistency_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Allocate diffusion, consistency, then interior FlowMap samples globally."""

    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("FlowMap interval samples must be aligned [B] tensors")
    batch = int(first.shape[0])
    start = int(global_start_index)
    total = int(global_batch_size)
    if start < 0 or total <= 0 or start + batch > total:
        raise ValueError("global batch indices are inconsistent")
    t = torch.maximum(first, second)
    r = torch.minimum(first, second)
    diffusion_count = round(float(diffusion_ratio) * total)
    consistency_count = round(float(consistency_ratio) * total)
    indices = torch.arange(start, start + batch, device=t.device)
    diffusion = indices < diffusion_count
    consistency = (indices >= diffusion_count) & (indices < diffusion_count + consistency_count)
    r = torch.where(diffusion, t, r)
    r = torch.where(consistency, torch.zeros_like(r), r)
    return t, r, diffusion


def flowmap_central_difference(
    prediction_plus: Tensor,
    prediction_minus: Tensor,
    *,
    epsilon: float,
    guidance_scale: float,
) -> Tensor:
    """Official central derivative in model timestep units."""

    if prediction_plus.shape != prediction_minus.shape:
        raise ValueError("central-difference predictions must have identical shapes")
    epsilon_value = float(epsilon)
    guidance = float(guidance_scale)
    if epsilon_value <= 0 or guidance <= 0:
        raise ValueError("central-difference epsilon and guidance must be positive")
    return (prediction_plus - prediction_minus) / (2.0 * epsilon_value * guidance)


def flowmap_target(
    clean: Tensor,
    noise: Tensor,
    derivative: Tensor,
    timesteps: Tensor,
    destination_timesteps: Tensor,
) -> Tensor:
    """Return ``(noise-clean) - (t-r)*dF/dt`` in upstream timestep units."""

    if not (clean.shape == noise.shape == derivative.shape):
        raise ValueError("FlowMap target tensors must have identical shapes")
    current = _time_coefficients(timesteps, clean).to(dtype=clean.dtype)
    destination = _time_coefficients(destination_timesteps, clean).to(dtype=clean.dtype)
    return (noise - clean) - (current - destination) * derivative


def fused_guidance_prediction(
    conditional: Tensor,
    unconditional: Tensor,
    guidance_scale: float,
) -> Tensor:
    """AnyFlow pretraining fusion ``(cond-(1-g)*uncond)/g``."""

    if conditional.shape != unconditional.shape:
        raise ValueError("guidance predictions must have identical shapes")
    scale = float(guidance_scale)
    if scale <= 0:
        raise ValueError("guidance scale must be positive")
    return (conditional - (1.0 - scale) * unconditional) / scale


def balance_flowmap_losses(
    losses: Tensor,
    diffusion_mask: Tensor,
    global_diffusion_mean: Tensor,
    *,
    epsilon: float = 1.0e-5,
) -> Tensor:
    """Match every non-diffusion loss to the global diffusion mean."""

    if losses.ndim != 1 or diffusion_mask.shape != losses.shape:
        raise ValueError("losses and diffusion_mask must be aligned [B] tensors")
    if diffusion_mask.dtype is not torch.bool:
        raise TypeError("diffusion_mask must be boolean")
    mean = torch.as_tensor(
        global_diffusion_mean,
        device=losses.device,
        dtype=losses.dtype,
    )
    if mean.numel() != 1 or not bool(torch.isfinite(mean)):
        raise ValueError("global diffusion mean must be one finite scalar")
    with torch.no_grad():
        scale = mean / (losses.detach() + float(epsilon))
    return torch.where(diffusion_mask, losses, losses * scale)


def anyflow_real_guidance(
    conditional_clean: Tensor,
    unconditional_clean: Tensor,
    guidance_scale: float,
) -> Tensor:
    """Official on-policy convention ``cond + (cond-uncond)*scale``."""

    if conditional_clean.shape != unconditional_clean.shape:
        raise ValueError("real-score guidance predictions must have identical shapes")
    return conditional_clean + (conditional_clean - unconditional_clean) * float(guidance_scale)


def anyflow_distribution_gradient(
    generated: Tensor,
    fake_clean: Tensor,
    real_clean: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-sample normalized DMD gradient, including official nan cleanup."""

    if not (generated.shape == fake_clean.shape == real_clean.shape):
        raise ValueError("AnyFlow DMD tensors must have identical shapes")
    axes = tuple(range(1, generated.ndim))
    normalizer = (generated - real_clean).abs().mean(dim=axes, keepdim=True)
    gradient = torch.nan_to_num((fake_clean - real_clean) / normalizer)
    return gradient, normalizer


def anyflow_dmd_proxy_loss(
    generated: Tensor,
    distribution_gradient: Tensor,
) -> Tensor:
    """Inject the stopped DMD gradient through an fp64 MSE proxy."""

    if generated.shape != distribution_gradient.shape:
        raise ValueError("generated samples and DMD gradient must have identical shapes")
    value = generated.double()
    target = (value - distribution_gradient.double()).detach()
    return (value - target).square().mean()


def sample_logit_normal_time(
    batch_size: int,
    *,
    device: torch.device | str,
    mean: float,
    std: float,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample ``sigmoid(N(mean,std))`` fake-score corruption times."""

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if float(std) <= 0:
        raise ValueError("logit-normal standard deviation must be positive")
    normal = torch.randn(
        (int(batch_size),),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return torch.sigmoid(normal * float(std) + float(mean))


__all__ = [
    "allocate_flowmap_intervals",
    "anyflow_distribution_gradient",
    "anyflow_dmd_proxy_loss",
    "anyflow_real_guidance",
    "balance_flowmap_losses",
    "beta08_train_weight",
    "flowmap_central_difference",
    "flowmap_inference_schedule",
    "flowmap_interpolate",
    "flowmap_step",
    "flowmap_target",
    "fused_guidance_prediction",
    "sample_logit_normal_time",
    "shift_flowmap_time",
]
