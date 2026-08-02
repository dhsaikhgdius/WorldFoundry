"""Rectified-flow operations executed by Causal Consistency Distillation.

Key formulas:
  - Flow corruption: x_t = (1 - sigma) * x_0 + sigma * eps
  - CFG velocity: v = v_uncond + w * (v_cond - v_uncond)
  - Adjacent Euler step: x_{t-dt} = x_t - dt * v, dt = (t - t_next) / num_train_steps

References:
  - CausVid (causal video diffusion): https://arxiv.org/abs/2412.07772
  - Self-Forcing: https://github.com/guandeh17/Self-Forcing
"""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor


def flow_corrupt(clean_latents: Tensor, noise: Tensor, sigma: float) -> Tensor:
    """Return ``(1-sigma) * clean + sigma * noise``."""

    if clean_latents.shape != noise.shape:
        raise ValueError("clean_latents and noise must have identical shapes")
    value = float(sigma)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("sigma must be finite and in [0,1]")
    return (1.0 - value) * clean_latents + value * noise


def classifier_free_velocity(
    conditional: Tensor,
    unconditional: Tensor,
    guidance_scale: float,
) -> Tensor:
    if conditional.shape != unconditional.shape:
        raise ValueError("conditional and unconditional velocities must match")
    scale = float(guidance_scale)
    if not isfinite(scale):
        raise ValueError("guidance_scale must be finite")
    return unconditional + scale * (conditional - unconditional)


def adjacent_flow_euler_step(
    latent: Tensor,
    velocity: Tensor,
    *,
    timestep: float,
    next_timestep: float,
    num_train_timesteps: int,
) -> Tensor:
    """Advance one adjacent teacher step on the descending flow timeline."""

    if latent.shape != velocity.shape:
        raise ValueError("latent and velocity must have identical shapes")
    if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
        raise ValueError("num_train_timesteps must be an integer >= 2")
    current = float(timestep)
    following = float(next_timestep)
    if not isfinite(current) or not isfinite(following) or not current > following >= 0:
        raise ValueError("adjacent timesteps must be finite and strictly descending")
    dt = (current - following) / float(num_train_timesteps)
    return latent - dt * velocity


def full_frame_timesteps(reference: Tensor, timestep: float, *, frame_dim: int) -> Tensor:
    """Expand one shared level to the [B,F] causal-model contract."""

    axis = frame_dim if frame_dim >= 0 else reference.ndim + frame_dim
    if axis <= 0 or axis >= reference.ndim:
        raise ValueError(f"frame_dim {frame_dim} is invalid for the latent tensor")
    value = float(timestep)
    if not isfinite(value) or value < 0:
        raise ValueError("timestep must be finite and non-negative")
    return torch.full(
        (reference.shape[0], reference.shape[axis]),
        value,
        device=reference.device,
        dtype=torch.float32,
    )


__all__ = [
    "adjacent_flow_euler_step",
    "classifier_free_velocity",
    "flow_corrupt",
    "full_frame_timesteps",
]
