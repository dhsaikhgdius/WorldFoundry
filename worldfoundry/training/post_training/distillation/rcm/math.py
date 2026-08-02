"""Exact source formulas used by native bidirectional and causal rCM.

Key formulas:
  - Bidirectional sCM tangent (normalized): target = -cos(t)*sqrt(1-(r*sin(t))^2)*(v_stop-v_teacher)
    - r*sin(t)*x - d/dt v; loss = ||v - stop(v) - target/||target||||
  - Causal sCM tangent: target = -(v_stop - v_teacher) - r * t * d/dt v
  - dCM discrete consistency: mean ||x0_student - x0_teacher||^2
  - DMD proxy: grad = (x0_fake - x0_real) / mean|x_gen - x0_real|; L = ||x_gen - stop(x_gen - grad)||^2
  - Fake-score regression: mean ||x_gen - x0_fake||^2 / sin(t)^2

References:
  - rCM: https://arxiv.org/abs/2510.08431
  - Causal-rCM: https://arxiv.org/abs/2606.25473
  - DMD: https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..consistency.math import (
    batch_coefficients,
    rf_to_trigflow_time,
    shift_rf_time,
)


def _sample_reduction_dimensions(value: torch.Tensor) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise TypeError("rCM latent values must be [B,...] tensors")
    return tuple(range(1, value.ndim))


def sample_discrete_trigflow_path(
    reference: torch.Tensor,
    *,
    total_steps: int,
    skipping_interval_steps: int,
    timestep_shift: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Sample the official dCM RF grid and return equal-SNR TrigFlow times."""

    if skipping_interval_steps <= 0 or total_steps <= skipping_interval_steps:
        raise ValueError("dCM requires 0 < skipping_interval_steps < total_steps")
    du = 1.0 / float(total_steps)
    u = torch.rand(
        (reference.shape[0],),
        device=reference.device,
        dtype=torch.float64,
        generator=generator,
    ) * (1.0 - skipping_interval_steps * du)
    result = []
    for index in range(skipping_interval_steps + 1):
        reverse_time = 1.0 - (u + index * du)
        result.append(rf_to_trigflow_time(shift_rf_time(reverse_time, timestep_shift)))
    return tuple(result)


def sample_discrete_rf_path(
    reference: torch.Tensor,
    *,
    total_steps: int,
    skipping_interval_steps: int,
    timestep_shift: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Sample the RF-domain dCM grid used by causal teacher forcing."""

    if skipping_interval_steps <= 0 or total_steps <= skipping_interval_steps:
        raise ValueError("dCM requires 0 < skipping_interval_steps < total_steps")
    du = 1.0 / float(total_steps)
    u = torch.rand(
        (reference.shape[0],),
        device=reference.device,
        dtype=torch.float64,
        generator=generator,
    ) * (1.0 - skipping_interval_steps * du)
    return tuple(
        shift_rf_time(1.0 - (u + index * du), timestep_shift)
        for index in range(skipping_interval_steps + 1)
    )


def bidirectional_scm_loss(
    current_velocity: torch.Tensor,
    stopped_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    directional_derivative: torch.Tensor,
    noisy_latents: torch.Tensor,
    trig_timesteps: torch.Tensor,
    *,
    warmup_ratio: float,
    normalization_constant: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous rCM target from the fixed official trainer."""

    tensors = (
        current_velocity,
        stopped_velocity,
        teacher_velocity,
        directional_derivative,
        noisy_latents,
    )
    if any(value.shape != current_velocity.shape for value in tensors[1:]):
        raise ValueError("continuous rCM tensors must have identical shapes")
    ratio = float(warmup_ratio)
    constant = float(normalization_constant)
    if not 0 <= ratio <= 1 or constant <= 0:
        raise ValueError("warmup_ratio must be in [0,1] and normalization_constant positive")
    cosine = batch_coefficients(torch.cos(trig_timesteps), noisy_latents)
    sine = batch_coefficients(torch.sin(trig_timesteps), noisy_latents)
    target = -cosine * torch.sqrt(1.0 - ratio**2 * sine.square()) * (
        stopped_velocity - teacher_velocity
    )
    target = target - (ratio * cosine * sine * noisy_latents + directional_derivative)
    dimensions = _sample_reduction_dimensions(target)
    bad = torch.isnan(target).flatten(start_dim=1).any(dim=1)
    bad = bad | torch.isnan(current_velocity).flatten(start_dim=1).any(dim=1)
    mask = bad.reshape(bad.shape[0], *((1,) * (target.ndim - 1)))
    target = torch.where(mask, torch.zeros((), device=target.device, dtype=target.dtype), target)
    current = torch.where(mask, torch.zeros((), device=current_velocity.device, dtype=current_velocity.dtype), current_velocity)
    stopped = torch.where(mask, torch.zeros((), device=stopped_velocity.device, dtype=stopped_velocity.dtype), stopped_velocity)
    norm = torch.linalg.vector_norm(target.double(), dim=dimensions, keepdim=True)
    normalized = target.double() / (norm + constant)
    per_sample = (current.double() - stopped.double() - normalized).square().sum(dim=dimensions)
    return per_sample.mean(), norm.squeeze(), bad


def causal_scm_loss(
    current_velocity: torch.Tensor,
    stopped_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    directional_derivative: torch.Tensor,
    rf_timesteps: torch.Tensor,
    *,
    warmup_ratio: float,
    normalization_constant: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-forcing continuous causal-rCM target."""

    if not (
        current_velocity.shape
        == stopped_velocity.shape
        == teacher_velocity.shape
        == directional_derivative.shape
    ):
        raise ValueError("causal sCM tensors must have identical shapes")
    ratio = float(warmup_ratio)
    if not 0 <= ratio <= 1 or normalization_constant <= 0:
        raise ValueError("invalid causal sCM warmup or normalization")
    time = batch_coefficients(rf_timesteps, current_velocity)
    target = -(stopped_velocity - teacher_velocity) - ratio * time * directional_derivative
    dimensions = _sample_reduction_dimensions(target)
    bad = torch.isnan(target).flatten(start_dim=1).any(dim=1)
    bad = bad | torch.isnan(current_velocity).flatten(start_dim=1).any(dim=1)
    mask = bad.reshape(bad.shape[0], *((1,) * (target.ndim - 1)))
    target = torch.where(mask, torch.zeros((), device=target.device, dtype=target.dtype), target)
    current = torch.where(mask, torch.zeros((), device=current_velocity.device, dtype=current_velocity.dtype), current_velocity)
    stopped = torch.where(mask, torch.zeros((), device=stopped_velocity.device, dtype=stopped_velocity.dtype), stopped_velocity)
    norm = torch.linalg.vector_norm(target.double(), dim=dimensions, keepdim=True)
    normalized = target.double() / (norm + float(normalization_constant))
    per_sample = (current.double() - stopped.double() - normalized).square().sum(dim=dimensions)
    return per_sample.mean(), norm.squeeze(), bad


def discrete_consistency_loss(
    current_clean: torch.Tensor,
    target_clean: torch.Tensor,
    *,
    causal_video_reduction: bool = False,
) -> torch.Tensor:
    """Return the official dCM sample mean for either execution family."""

    if current_clean.shape != target_clean.shape:
        raise ValueError("dCM current and target predictions must match")
    if causal_video_reduction:
        if current_clean.ndim != 5:
            raise ValueError("causal video dCM expects [B,C,T,H,W]")
        per_sample = (current_clean - target_clean).square().mean(dim=(1, 3, 4)).sum(dim=1)
    else:
        per_sample = (current_clean - target_clean).square().sum(
            dim=_sample_reduction_dimensions(current_clean)
        )
    return per_sample.mean()


def exact_dmd_proxy_loss(
    generated_clean: torch.Tensor,
    fake_score_clean: torch.Tensor,
    teacher_clean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official per-sample fp64 DMD normalizer and no-half proxy loss."""

    if not (generated_clean.shape == fake_score_clean.shape == teacher_clean.shape):
        raise ValueError("DMD clean predictions must have identical shapes")
    dimensions = _sample_reduction_dimensions(generated_clean)
    generated = generated_clean.double()
    fake = fake_score_clean.double()
    teacher = teacher_clean.double()
    denominator = (generated - teacher).abs().mean(dim=dimensions, keepdim=True).clamp_min(1e-5)
    distribution_gradient = (fake - teacher) / denominator
    squared = (generated - (generated - distribution_gradient).detach()).square()
    bad = torch.isnan(squared).flatten(start_dim=1).any(dim=1)
    mask = bad.reshape(bad.shape[0], *((1,) * (squared.ndim - 1)))
    squared = torch.where(mask, torch.zeros((), device=squared.device, dtype=squared.dtype), squared)
    return squared.sum(dim=dimensions).mean(), denominator, bad


def trigflow_fake_score_loss(
    generated_clean: torch.Tensor,
    fake_score_clean: torch.Tensor,
    trig_timesteps: torch.Tensor,
) -> torch.Tensor:
    """Official bidirectional rCM fake-score regression."""

    if generated_clean.shape != fake_score_clean.shape:
        raise ValueError("fake-score clean prediction must match generated clean")
    sine = batch_coefficients(torch.sin(trig_timesteps), generated_clean)
    per_sample = ((generated_clean - fake_score_clean).square() / sine.square()).sum(
        dim=_sample_reduction_dimensions(generated_clean)
    )
    return per_sample.mean()


def sum_scaled_losses(losses: Sequence[tuple[torch.Tensor, float]]) -> torch.Tensor:
    """Combine enabled objective terms without creating a dead zero tensor."""

    enabled = [(loss, float(scale)) for loss, scale in losses if float(scale) > 0]
    if not enabled:
        raise ValueError("at least one objective term must be enabled")
    total = enabled[0][0] * enabled[0][1]
    for loss, scale in enabled[1:]:
        total = total + loss * scale
    return total


__all__ = [
    "bidirectional_scm_loss",
    "causal_scm_loss",
    "discrete_consistency_loss",
    "exact_dmd_proxy_loss",
    "sample_discrete_rf_path",
    "sample_discrete_trigflow_path",
    "sum_scaled_losses",
    "trigflow_fake_score_loss",
]
