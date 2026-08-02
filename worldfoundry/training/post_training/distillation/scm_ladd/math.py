"""Pure SANA-Sprint transformation, consistency, and adversarial formulas.

Key formulas:
  - TrigFlow -> flow time (Eq. 6): t_flow = sin(t) / (sin(t) + cos(t))
  - Flow -> TrigFlow velocity (Eq. 8): v_trig = (linear*x + velocity*v_flow) / scale
  - sCM tangent (Alg. 2): g = -cos^2(t)*(sigma*v - v_teacher) - r*(cos*sin*x + sigma*d/dt v)
  - sCM loss: w = 1/tan(t); L = w * exp(-log_var) * ||v - stop(v) - g/||g||||^2 + log_var
  - LADD hinge: L_G = -D(fake); L_D = 0.5*(relu(1-D(real)) + relu(1+D(fake)))

References:
  - SANA-Sprint (sCM + LADD): https://arxiv.org/abs/2503.09641
  - rCM / TrigFlow parameterization: https://arxiv.org/abs/2510.08431
"""

from __future__ import annotations

from math import isfinite, pi

import torch
import torch.nn.functional as F

from ..consistency.math import batch_coefficients as _batch_coefficients
from ..consistency.math import (
    classifier_free_guidance,
    trigflow_clean_prediction,
    trigflow_interpolate,
)


def trigflow_to_flow_time(trig_timesteps: torch.Tensor) -> torch.Tensor:
    """SANA-Sprint Eq. 6: preserve SNR while mapping TrigFlow to flow time."""

    if not isinstance(trig_timesteps, torch.Tensor) or not trig_timesteps.is_floating_point():
        raise TypeError("trig_timesteps must be a floating-point tensor")
    if not bool(torch.isfinite(trig_timesteps).all()):
        raise ValueError("trig_timesteps must be finite")
    if bool(((trig_timesteps < 0) | (trig_timesteps > pi / 2)).any()):
        raise ValueError("TrigFlow timesteps must be in [0,pi/2]")
    sine = torch.sin(trig_timesteps)
    return sine / (sine + torch.cos(trig_timesteps))


def trigflow_to_flow_input(
    scaled_trig_latents: torch.Tensor,
    trig_timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SANA-Sprint Eq. 7: return flow input, flow time, and scale."""

    if not isinstance(scaled_trig_latents, torch.Tensor) or scaled_trig_latents.ndim < 2:
        raise TypeError("scaled_trig_latents must be a [B,...] tensor")
    flow_time = trigflow_to_flow_time(trig_timesteps)
    scale = torch.sqrt(flow_time.square() + (1.0 - flow_time).square())
    return scaled_trig_latents * _batch_coefficients(scale, scaled_trig_latents), flow_time, scale


def flow_velocity_to_trigflow(
    flow_latents: torch.Tensor,
    flow_velocity: torch.Tensor,
    flow_timesteps: torch.Tensor,
) -> torch.Tensor:
    """SANA-Sprint Eq. 8: transform a flow velocity into TrigFlow velocity."""

    if flow_latents.shape != flow_velocity.shape:
        raise ValueError("flow_latents and flow_velocity must match")
    if flow_timesteps.ndim != 1 or flow_timesteps.shape[0] != flow_latents.shape[0]:
        raise ValueError("flow_timesteps must have shape [B]")
    scale = torch.sqrt(flow_timesteps.square() + (1.0 - flow_timesteps).square())
    linear = 1.0 - 2.0 * flow_timesteps
    velocity = 1.0 - 2.0 * flow_timesteps + 2.0 * flow_timesteps.square()
    return (
        _batch_coefficients(linear, flow_latents) * flow_latents
        + _batch_coefficients(velocity, flow_latents) * flow_velocity
    ) / _batch_coefficients(scale, flow_latents)


def classifier_free_velocity(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guidance_scales: torch.Tensor,
) -> torch.Tensor:
    """Apply the classifier-free velocity rule used by the fixed Sana trainer."""

    return classifier_free_guidance(conditional, unconditional, guidance_scales)


def scm_tangent_target(
    noisy_latents: torch.Tensor,
    stopped_velocity: torch.Tensor,
    velocity_directional_derivative: torch.Tensor,
    teacher_path_velocity: torch.Tensor,
    trig_timesteps: torch.Tensor,
    *,
    sigma_data: float,
    warmup_ratio: float,
    normalization_constant: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SANA-Sprint Algorithm 2 lines 18-19 JVP rearrangement."""

    if not (
        noisy_latents.shape
        == stopped_velocity.shape
        == velocity_directional_derivative.shape
        == teacher_path_velocity.shape
    ):
        raise ValueError("all SCM tangent tensors must match")
    ratio = float(warmup_ratio)
    constant = float(normalization_constant)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0,1]")
    if constant <= 0:
        raise ValueError("normalization_constant must be positive")
    cosine = _batch_coefficients(torch.cos(trig_timesteps), noisy_latents)
    sine = _batch_coefficients(torch.sin(trig_timesteps), noisy_latents)
    tangent = -cosine.square() * (float(sigma_data) * stopped_velocity - teacher_path_velocity)
    tangent = tangent - ratio * (
        cosine * sine * noisy_latents
        + float(sigma_data) * velocity_directional_derivative
    )
    dimensions = tuple(range(1, tangent.ndim))
    norm = torch.linalg.vector_norm(tangent.float(), dim=dimensions, keepdim=True)
    return tangent / (norm.to(tangent.dtype) + constant), norm


def scm_adaptive_loss(
    current_velocity: torch.Tensor,
    stopped_velocity: torch.Tensor,
    tangent_target: torch.Tensor,
    log_variance: torch.Tensor,
    trig_timesteps: torch.Tensor,
    *,
    sigma_data: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official sCM weighted objective, including learned log-variance."""

    if not (current_velocity.shape == stopped_velocity.shape == tangent_target.shape):
        raise ValueError("SCM velocity and tangent tensors must match")
    if log_variance.ndim == 1:
        log_variance = _batch_coefficients(log_variance, current_velocity)
    try:
        torch.broadcast_shapes(log_variance.shape, current_velocity.shape)
    except RuntimeError as error:
        raise ValueError("log_variance cannot broadcast to the velocity tensor") from error
    sigma = torch.tan(trig_timesteps) * float(sigma_data)
    if bool((sigma <= 0).any()):
        raise ValueError("sCM timesteps must be strictly greater than zero")
    weight = _batch_coefficients(sigma.reciprocal(), current_velocity)
    squared = (current_velocity - stopped_velocity - tangent_target).square()
    weighted = weight * torch.exp(-log_variance) * squared + log_variance
    return weighted.mean(), (weight * squared).mean(), squared.mean()


def ladd_generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(fake_logits, torch.Tensor) or fake_logits.numel() == 0:
        raise TypeError("fake_logits must be a non-empty tensor")
    return -fake_logits.float().mean()


def ladd_discriminator_hinge_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(real_logits, torch.Tensor) or not isinstance(fake_logits, torch.Tensor):
        raise TypeError("LADD logits must be tensors")
    if real_logits.numel() == 0 or fake_logits.numel() == 0:
        raise ValueError("LADD logits cannot be empty")
    real_loss = F.relu(1.0 - real_logits.float()).mean()
    fake_loss = F.relu(1.0 + fake_logits.float()).mean()
    return 0.5 * (real_loss + fake_loss), real_loss, fake_loss


def sample_trigflow_timesteps(
    reference: torch.Tensor,
    *,
    logit_mean: float,
    logit_std: float,
    sigma_data: float,
    max_time_probability: float = 0.0,
    max_time: float = pi / 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Algorithm 2 log-normal TrigFlow time, optionally mixed with max time."""

    if not isinstance(reference, torch.Tensor) or reference.ndim < 2 or reference.shape[0] == 0:
        raise TypeError("reference must be a non-empty [B,...] tensor")
    standard_deviation = float(logit_std)
    data_scale = float(sigma_data)
    probability = float(max_time_probability)
    largest = float(max_time)
    if not all(isfinite(value) for value in (float(logit_mean), standard_deviation, data_scale, probability, largest)):
        raise ValueError("TrigFlow sampler values must be finite")
    if standard_deviation <= 0 or data_scale <= 0:
        raise ValueError("logit_std and sigma_data must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("max_time_probability must be in [0,1]")
    if not 0.0 < largest <= 1.57080:
        raise ValueError("max_time must be in (0,1.57080]")
    tau = torch.randn(
        (reference.shape[0],),
        device=reference.device,
        dtype=torch.float32,
        generator=generator,
    )
    tau = tau * standard_deviation + float(logit_mean)
    timesteps = torch.atan(torch.exp(tau) / data_scale)
    if probability:
        selected = torch.rand(
            timesteps.shape,
            device=timesteps.device,
            dtype=timesteps.dtype,
            generator=generator,
        ) < probability
        timesteps = torch.where(selected, torch.full_like(timesteps, largest), timesteps)
    return timesteps


def _normal_like(reference: torch.Tensor, *, sigma_data: float, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    ) * float(sigma_data)


__all__ = [
    "classifier_free_velocity",
    "flow_velocity_to_trigflow",
    "ladd_discriminator_hinge_loss",
    "ladd_generator_hinge_loss",
    "sample_trigflow_timesteps",
    "scm_adaptive_loss",
    "scm_tangent_target",
    "trigflow_clean_prediction",
    "trigflow_interpolate",
    "trigflow_to_flow_input",
    "trigflow_to_flow_time",
]
