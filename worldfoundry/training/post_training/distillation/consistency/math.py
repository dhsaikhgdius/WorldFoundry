"""Parameterization math shared by sCM-LADD and rCM.

Key formulas:
  - TrigFlow interpolation: x_t = cos(t) * x_0 + sin(t) * eps
  - TrigFlow clean recovery: x_0 = cos(t) * x_t - sin(t) * sigma_data * v
  - RF timestep shift: t' = shift * t / (1 + (shift - 1) * t)
  - Equal-SNR RF <-> TrigFlow: t_trig = atan(t_rf / (1 - t_rf))
  - CFG: y = u + w * (c - u)
  - LogNormal RF time: t = sigmoid(N(mean, std))

References:
  - rCM (Rectified Consistency Model): https://arxiv.org/abs/2510.08431
  - Executable oracle: https://github.com/NVlabs/rCM (commit ed3cb14)
"""

from __future__ import annotations

from math import isfinite

import torch


def batch_coefficients(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Expand one scalar per batch item over a latent tensor."""

    if not isinstance(values, torch.Tensor) or values.ndim not in {1, 2}:
        raise TypeError("timestep coefficients must be a [B] or [B,1] tensor")
    if values.ndim == 2:
        if values.shape[1] != 1:
            raise ValueError("two-dimensional timestep coefficients must have shape [B,1]")
        values = values[:, 0]
    if values.shape[0] != reference.shape[0]:
        raise ValueError("timestep coefficients and reference batch sizes must match")
    return values.reshape(values.shape[0], *((1,) * (reference.ndim - 1)))


def trigflow_interpolate(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    trig_timesteps: torch.Tensor,
) -> torch.Tensor:
    """Return ``cos(t) * x0 + sin(t) * noise``."""

    if clean_latents.shape != noise.shape:
        raise ValueError("clean_latents and noise must match")
    cosine = batch_coefficients(torch.cos(trig_timesteps), clean_latents)
    sine = batch_coefficients(torch.sin(trig_timesteps), clean_latents)
    return cosine * clean_latents + sine * noise


def trigflow_clean_prediction(
    noisy_latents: torch.Tensor,
    trig_velocity: torch.Tensor,
    trig_timesteps: torch.Tensor,
    *,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    """Recover clean data from a TrigFlow velocity prediction."""

    if noisy_latents.shape != trig_velocity.shape:
        raise ValueError("noisy_latents and trig_velocity must match")
    scale = float(sigma_data)
    if not isfinite(scale) or scale <= 0:
        raise ValueError("sigma_data must be finite and positive")
    cosine = batch_coefficients(torch.cos(trig_timesteps), noisy_latents)
    sine = batch_coefficients(torch.sin(trig_timesteps), noisy_latents)
    return cosine * noisy_latents - sine * scale * trig_velocity


def classifier_free_guidance(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guidance_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Apply ``uncond + scale * (cond - uncond)`` without changing shape."""

    if conditional.shape != unconditional.shape:
        raise ValueError("conditional and unconditional predictions must match")
    if isinstance(guidance_scale, torch.Tensor):
        scale = batch_coefficients(guidance_scale, conditional)
    else:
        value = float(guidance_scale)
        if not isfinite(value):
            raise ValueError("guidance_scale must be finite")
        scale = value
    return unconditional + scale * (conditional - unconditional)


def shift_rf_time(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the standard rectified-flow timestep shift."""

    resolved = float(shift)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError("shift must be finite and positive")
    if not isinstance(timesteps, torch.Tensor) or not timesteps.is_floating_point():
        raise TypeError("timesteps must be a floating-point tensor")
    if bool(((timesteps < 0) | (timesteps > 1)).any()):
        raise ValueError("rectified-flow timesteps must be in [0,1]")
    return resolved * timesteps / (1.0 + (resolved - 1.0) * timesteps)


def rf_to_trigflow_time(timesteps: torch.Tensor) -> torch.Tensor:
    """Map RF time to the equal-SNR TrigFlow angle."""

    if not isinstance(timesteps, torch.Tensor) or not timesteps.is_floating_point():
        raise TypeError("timesteps must be a floating-point tensor")
    if bool(((timesteps < 0) | (timesteps > 1)).any()):
        raise ValueError("rectified-flow timesteps must be in [0,1]")
    denominator = (1.0 - timesteps).clamp_min(torch.finfo(timesteps.dtype).tiny)
    angle = torch.atan(timesteps / denominator)
    return torch.where(timesteps == 1, torch.full_like(angle, torch.pi / 2), angle)


def trigflow_to_rf_time(timesteps: torch.Tensor) -> torch.Tensor:
    """Map a TrigFlow angle to the equal-SNR rectified-flow time."""

    if not isinstance(timesteps, torch.Tensor) or not timesteps.is_floating_point():
        raise TypeError("timesteps must be a floating-point tensor")
    half_pi = torch.pi / 2
    if bool(((timesteps < 0) | (timesteps > half_pi)).any()):
        raise ValueError("TrigFlow timesteps must be in [0,pi/2]")
    sine = torch.sin(timesteps)
    return sine / (torch.cos(timesteps) + sine)


def sample_lognormal_rf_time(
    reference: torch.Tensor,
    *,
    mean: float,
    std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the rCM LogNormal RF distribution as ``sigmoid(N(mean,std))``."""

    if not isinstance(reference, torch.Tensor) or reference.ndim < 2 or reference.shape[0] == 0:
        raise TypeError("reference must be a non-empty [B,...] tensor")
    resolved_mean = float(mean)
    resolved_std = float(std)
    if not isfinite(resolved_mean) or not isfinite(resolved_std) or resolved_std <= 0:
        raise ValueError("lognormal mean must be finite and std must be positive")
    normal = torch.randn(
        (reference.shape[0],),
        device=reference.device,
        dtype=torch.float64,
        generator=generator,
    )
    return torch.sigmoid(normal * resolved_std + resolved_mean)


__all__ = [
    "batch_coefficients",
    "classifier_free_guidance",
    "rf_to_trigflow_time",
    "sample_lognormal_rf_time",
    "shift_rf_time",
    "trigflow_clean_prediction",
    "trigflow_interpolate",
    "trigflow_to_rf_time",
]
