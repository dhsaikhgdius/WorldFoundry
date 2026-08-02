"""Pure SGMD formula components.

Key formulas:
  - Normalized Fisher (teacher stop-grad): L_F = 0.5 * ||x0_fake - x0_teacher||^2 / mean|x_gen - x0_teacher|
  - Fake correction (RC inner loop): pseudo = x0_fake - (x0_fake - x_gen)/sigma; L = 0.5 * ||x0_fake - pseudo||^2
  - Fake-score flow: L = 0.5 * ||v_pred - (x_gen - eps)/sigma||^2
  - Diversity: target = (eps - x_anchor) / (1 - sigma); L = ||v_student - target||^2

References:
  - SGMD (Score Gradient Matching Distillation): https://arxiv.org/abs/2605.30116
  - DMD2: https://arxiv.org/abs/2405.14867
"""

from __future__ import annotations

from math import isfinite

import torch

from worldfoundry.training.objectives.flow_matching import (
    flow_clean_from_velocity,
    flow_interpolate,
    flow_shift_sigmas,
    flow_velocity_target,
)


def _matching_tensors(*values: torch.Tensor) -> None:
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("SGMD formula inputs must be torch.Tensor values")
    if values[0].ndim < 2 or any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("SGMD latent tensors must have one matching [B,...] shape")


def expand_sigmas(sigmas: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(sigmas, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError("sigmas and reference must be torch.Tensor values")
    if reference.ndim < 2:
        raise ValueError("reference must include batch and feature dimensions")
    if sigmas.ndim == 0:
        sigmas = sigmas.expand(reference.shape[0])
    if sigmas.ndim != 1 or sigmas.shape[0] != reference.shape[0]:
        raise ValueError("sigmas must be scalar or have shape [B]")
    return sigmas.reshape((sigmas.shape[0],) + (1,) * (reference.ndim - 1))


def sgmd_classifier_free_guidance(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    _matching_tensors(unconditional, conditional)
    scale = float(guidance_scale)
    if not isfinite(scale) or scale < 0:
        raise ValueError("guidance_scale must be finite and non-negative")
    return unconditional + scale * (conditional - unconditional)


def sgmd_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigmas: torch.Tensor,
    next_sigmas: torch.Tensor,
) -> torch.Tensor:
    _matching_tensors(sample, velocity)
    sigma = expand_sigmas(sigmas, sample)
    sigma_next = expand_sigmas(next_sigmas, sample)
    return (sample + (sigma_next - sigma) * velocity).to(dtype=sample.dtype)


def sgmd_normalized_fisher_loss_per_sample(
    generated: torch.Tensor,
    fake_clean: torch.Tensor,
    teacher_clean: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    _matching_tensors(generated, fake_clean, teacher_clean)
    resolved_epsilon = float(epsilon)
    if not isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    axes = tuple(range(1, generated.ndim))
    with torch.no_grad():
        normalizer = (
            generated.float() - teacher_clean.float()
        ).abs().mean(dim=axes, keepdim=True)
    per_element = (
        0.5
        * (fake_clean.float() - teacher_clean.float()).square()
        / (normalizer + resolved_epsilon)
    )
    return per_element.flatten(1).mean(1), normalizer


def sgmd_fake_correction_loss_per_sample(
    generated: torch.Tensor,
    fake_clean: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    _matching_tensors(generated, fake_clean)
    resolved_epsilon = float(epsilon)
    if not isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    expanded = expand_sigmas(sigmas.float(), fake_clean).clamp_min(resolved_epsilon)
    with torch.no_grad():
        gradient = (fake_clean.float() - generated.detach().float()) / expanded
        pseudo_target = fake_clean.float() - gradient
    return 0.5 * (fake_clean.float() - pseudo_target).square().flatten(1).mean(1)


def sgmd_fake_score_flow_loss_per_sample(
    velocity_prediction: torch.Tensor,
    generated: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    _matching_tensors(velocity_prediction, generated, noise)
    target = flow_velocity_target(generated.float(), noise.float()).detach()
    return 0.5 * (velocity_prediction.float() - target).square().flatten(1).mean(1)


def sgmd_fake_clean_diagnostic_per_sample(
    generated: torch.Tensor,
    fake_clean: torch.Tensor,
    normalizer: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    _matching_tensors(generated, fake_clean)
    try:
        torch.broadcast_shapes(fake_clean.shape, normalizer.shape)
    except RuntimeError as error:
        raise ValueError("normalizer cannot broadcast to fake_clean") from error
    return (
        0.5
        * (fake_clean.float() - generated.float()).square()
        / (normalizer.float() + float(epsilon))
    ).flatten(1).mean(1)


def sgmd_diversity_loss_per_sample(
    initial_noise: torch.Tensor,
    anchor_latent: torch.Tensor,
    anchor_sigmas: torch.Tensor,
    student_velocity: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    _matching_tensors(initial_noise, anchor_latent, student_velocity)
    denominator = (1.0 - expand_sigmas(anchor_sigmas.float(), initial_noise)).clamp_min(
        float(epsilon)
    )
    with torch.no_grad():
        target = (initial_noise.float() - anchor_latent.float()) / denominator
    return (student_velocity.float() - target).square().flatten(1).mean(1)


__all__ = [
    "expand_sigmas",
    "flow_clean_from_velocity",
    "flow_interpolate",
    "flow_shift_sigmas",
    "sgmd_classifier_free_guidance",
    "sgmd_diversity_loss_per_sample",
    "sgmd_euler_step",
    "sgmd_fake_clean_diagnostic_per_sample",
    "sgmd_fake_correction_loss_per_sample",
    "sgmd_fake_score_flow_loss_per_sample",
    "sgmd_normalized_fisher_loss_per_sample",
]
