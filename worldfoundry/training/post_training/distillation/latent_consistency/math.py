"""Model-independent latent consistency and deterministic DDIM primitives.

Key formulas:
  - Forward diffusion: x_t = alpha_t * x_0 + sigma_t * eps
  - Boundary condition: x_pred = c_skip(t) * x_t + c_out(t) * x_0
    with c_skip = sigma_data^2 / (s^2 + sigma_data^2), c_out = s / sqrt(s^2 + sigma_data^2)
  - DDIM (eta=0): x_{t-1} = sqrt(alpha_{t-1}) * x_0 + sqrt(1-alpha_{t-1}) * eps
  - LCM CFG: y = c + w * (c - u)
  - Robust loss: L2 = ||delta||^2; pseudo-Huber = sqrt(delta^2 + c^2) - c

References:
  - Latent Consistency Models (LCM): https://arxiv.org/abs/2310.04378
"""

from __future__ import annotations

from math import log

import torch
from torch import Tensor

from .config import LatentConsistencyPredictionType


def append_dims(value: Tensor, target_ndim: int) -> Tensor:
    """Append singleton dimensions for batch-wise latent coefficients."""

    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if isinstance(target_ndim, bool) or not isinstance(target_ndim, int):
        raise TypeError("target_ndim must be an integer")
    if target_ndim < value.ndim:
        raise ValueError("target_ndim cannot be smaller than value.ndim")
    return value[(...,) + (None,) * (target_ndim - value.ndim)]


def gather_schedule_coefficients(
    schedule: Tensor,
    timesteps: Tensor,
    reference: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Gather one scalar schedule coefficient per latent sample."""

    if not isinstance(schedule, Tensor) or schedule.ndim != 1:
        raise TypeError("schedule must be a one-dimensional tensor")
    if not isinstance(timesteps, Tensor) or timesteps.ndim != 1:
        raise TypeError("timesteps must be a one-dimensional tensor")
    if timesteps.dtype != torch.int64:
        raise TypeError("timesteps must have dtype torch.int64")
    if not isinstance(reference, Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    if timesteps.shape[0] != reference.shape[0]:
        raise ValueError("timesteps must match the reference batch size")
    values = schedule.to(device=reference.device).gather(0, timesteps)
    if dtype is not None:
        values = values.to(dtype=dtype)
    return append_dims(values, reference.ndim)


def guidance_scale_embedding(
    guidance_coefficients: Tensor,
    *,
    embedding_dim: int,
    embedding_scale: float = 1000.0,
    max_period: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sinusoidally embed the LCM guidance coefficient."""

    if not isinstance(guidance_coefficients, Tensor) or guidance_coefficients.ndim != 1:
        raise TypeError("guidance_coefficients must be a one-dimensional tensor")
    if not guidance_coefficients.is_floating_point():
        raise TypeError("guidance_coefficients must be floating point")
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim < 4:
        raise ValueError("embedding_dim must be an integer of at least four")
    scale = float(embedding_scale)
    period = float(max_period)
    if not 0 < scale < float("inf") or not 1 < period < float("inf"):
        raise ValueError("embedding_scale and max_period must be finite positive values")
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        torch.arange(
            half_dim,
            device=guidance_coefficients.device,
            dtype=torch.float32,
        )
        * (-log(period) / (half_dim - 1))
    )
    angles = guidance_coefficients.float()[:, None] * scale * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding_dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding.to(dtype=dtype)


def boundary_condition_scalings(
    timesteps: Tensor,
    *,
    sigma_data: float = 0.5,
    timestep_scaling: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Return LCM skip/output coefficients with an exact identity at time zero."""

    if not isinstance(timesteps, Tensor) or timesteps.ndim != 1:
        raise TypeError("timesteps must be a one-dimensional tensor")
    scaled = timesteps.float() * float(timestep_scaling)
    sigma_squared = float(sigma_data) ** 2
    denominator = scaled.square() + sigma_squared
    return sigma_squared / denominator, scaled / denominator.sqrt()


def add_forward_diffusion_noise(
    clean_latents: Tensor,
    noise: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Apply the DDPM forward process at a batch of discrete timesteps."""

    if clean_latents.shape != noise.shape:
        raise ValueError("noise must match clean_latents")
    if clean_latents.device != noise.device or clean_latents.dtype != noise.dtype:
        raise ValueError("noise must share clean_latents device and dtype")
    return alpha.to(dtype=clean_latents.dtype) * clean_latents + sigma.to(dtype=clean_latents.dtype) * noise


def prediction_to_origin_and_epsilon(
    model_output: Tensor,
    noisy_latents: Tensor,
    alpha: Tensor,
    sigma: Tensor,
    *,
    prediction_type: LatentConsistencyPredictionType,
) -> tuple[Tensor, Tensor]:
    """Resolve both x0 and epsilon from epsilon- or velocity-prediction models."""

    if model_output.shape != noisy_latents.shape:
        raise ValueError("model_output must match noisy_latents")
    if prediction_type == "epsilon":
        epsilon = model_output
        origin = (noisy_latents - sigma * epsilon) / alpha
    elif prediction_type == "v_prediction":
        origin = alpha * noisy_latents - sigma * model_output
        epsilon = alpha * model_output + sigma * noisy_latents
    else:
        raise ValueError(f"unsupported prediction_type: {prediction_type!r}")
    return origin, epsilon


def classifier_free_guidance(
    conditional: Tensor,
    unconditional: Tensor,
    guidance_coefficients: Tensor,
) -> Tensor:
    """Apply the coefficient convention used by LCM training: c + w(c-u)."""

    if conditional.shape != unconditional.shape:
        raise ValueError("conditional and unconditional predictions must match")
    if guidance_coefficients.ndim != 1 or guidance_coefficients.shape[0] != conditional.shape[0]:
        raise ValueError("guidance_coefficients must have shape [B]")
    weights = append_dims(
        guidance_coefficients.to(device=conditional.device, dtype=conditional.dtype),
        conditional.ndim,
    )
    return conditional + weights * (conditional - unconditional)


def deterministic_ddim_step(
    predicted_origin: Tensor,
    predicted_epsilon: Tensor,
    previous_alpha_cumprod: Tensor,
) -> Tensor:
    """Take one eta-zero DDIM step to the preceding distillation level."""

    if predicted_origin.shape != predicted_epsilon.shape:
        raise ValueError("DDIM origin and epsilon predictions must match")
    alpha = previous_alpha_cumprod.sqrt()
    sigma = (1.0 - previous_alpha_cumprod).clamp_min(0.0).sqrt()
    return alpha * predicted_origin + sigma * predicted_epsilon


def consistency_prediction(
    noisy_latents: Tensor,
    predicted_origin: Tensor,
    c_skip: Tensor,
    c_out: Tensor,
) -> Tensor:
    """Apply the discrete LCM boundary-condition parameterization."""

    if noisy_latents.shape != predicted_origin.shape:
        raise ValueError("predicted_origin must match noisy_latents")
    return c_skip * noisy_latents + c_out * predicted_origin


def latent_consistency_elementwise_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_type: str,
    pseudo_huber_c: float | None,
) -> Tensor:
    """Return the released elementwise L2 or Charbonnier-style robust loss."""

    if prediction.shape != target.shape:
        raise ValueError("latent consistency prediction and target must match")
    difference = prediction.float() - target.float()
    if loss_type == "l2":
        return difference.square()
    if loss_type == "pseudo_huber":
        if pseudo_huber_c is None:
            raise ValueError("pseudo_huber loss requires pseudo_huber_c")
        c = float(pseudo_huber_c)
        return (difference.square() + c * c).sqrt() - c
    raise ValueError(f"unsupported latent consistency loss_type: {loss_type!r}")


__all__ = [
    "add_forward_diffusion_noise",
    "append_dims",
    "boundary_condition_scalings",
    "classifier_free_guidance",
    "consistency_prediction",
    "deterministic_ddim_step",
    "gather_schedule_coefficients",
    "guidance_scale_embedding",
    "latent_consistency_elementwise_loss",
    "prediction_to_origin_and_epsilon",
]
