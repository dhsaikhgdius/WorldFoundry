"""Official progressive-distillation schedule and target equations.

Key formulas:
  - Cosine log-SNR: lambda(t) = -2 log tan(t * span + offset)
  - alpha/sigma from log-SNR: alpha = sqrt(sigmoid(lambda)), sigma = sqrt(sigmoid(-lambda))
  - Forward: x_t = alpha * x_0 + sigma * eps
  - DDIM step: x_{target} = alpha_target * x_0 + sigma_target * eps
  - Implied clean target from two teacher steps (Salimans & Ho)
  - Loss modes: constant -> MSE(x_0); snr -> MSE(eps); snr_trunc -> max(MSE(x_0), MSE(eps))

References:
  - Progressive Distillation: https://arxiv.org/abs/2202.00512
"""

from __future__ import annotations

from math import atan, exp, isfinite

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ProgressiveLossWeight, ProgressivePredictionType


def append_dims(value: Tensor, target_ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if isinstance(target_ndim, bool) or not isinstance(target_ndim, int):
        raise TypeError("target_ndim must be an integer")
    if value.ndim > target_ndim:
        raise ValueError("target_ndim cannot be smaller than value.ndim")
    return value[(...,) + (None,) * (target_ndim - value.ndim)]


def cosine_logsnr(
    times: Tensor,
    *,
    logsnr_min: float,
    logsnr_max: float,
) -> Tensor:
    """Google Research cosine schedule with exact configured endpoints."""

    if not isinstance(times, Tensor) or not times.is_floating_point():
        raise TypeError("times must be a floating-point tensor")
    if bool(((times < 0) | (times > 1)).any()):
        raise ValueError("times must lie in [0,1]")
    minimum = float(logsnr_min)
    maximum = float(logsnr_max)
    if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
        raise ValueError("logsnr bounds must be finite and strictly ordered")
    offset = atan(exp(-0.5 * maximum))
    span = atan(exp(-0.5 * minimum)) - offset
    return -2.0 * torch.log(torch.tan(times * span + offset))


def alpha_sigma(logsnr: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(logsnr, Tensor) or logsnr.ndim != 1:
        raise TypeError("logsnr must have shape [B]")
    if logsnr.shape[0] != reference.shape[0]:
        raise ValueError("logsnr and reference batch sizes differ")
    alpha = append_dims(torch.sigmoid(logsnr).sqrt(), reference.ndim)
    sigma = append_dims(torch.sigmoid(-logsnr).sqrt(), reference.ndim)
    return alpha, sigma


def add_forward_noise(
    clean: Tensor,
    noise: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    if clean.shape != noise.shape or clean.device != noise.device:
        raise ValueError("clean and noise tensors must match")
    return alpha.to(dtype=clean.dtype) * clean + sigma.to(dtype=clean.dtype) * noise


def prediction_to_clean_epsilon_velocity(
    model_output: Tensor,
    noisy: Tensor,
    alpha: Tensor,
    sigma: Tensor,
    *,
    prediction_type: ProgressivePredictionType,
) -> tuple[Tensor, Tensor, Tensor]:
    if model_output.shape != noisy.shape:
        raise ValueError("model output must match the noisy latent shape")
    if prediction_type == "sample":
        clean = model_output
        epsilon = (noisy - alpha * clean) / sigma
    elif prediction_type == "epsilon":
        epsilon = model_output
        clean = (noisy - sigma * epsilon) / alpha
    elif prediction_type == "v_prediction":
        clean = alpha * noisy - sigma * model_output
        epsilon = alpha * model_output + sigma * noisy
    else:
        raise ValueError(f"unsupported prediction_type: {prediction_type!r}")
    velocity = alpha * epsilon - sigma * clean
    return clean, epsilon, velocity


def deterministic_ddim_step(
    predicted_clean: Tensor,
    predicted_epsilon: Tensor,
    target_logsnr: Tensor,
) -> Tensor:
    if predicted_clean.shape != predicted_epsilon.shape:
        raise ValueError("DDIM clean and epsilon predictions must match")
    alpha, sigma = alpha_sigma(target_logsnr, predicted_clean)
    return alpha * predicted_clean + sigma * predicted_epsilon


def implied_clean_target(
    noisy_start: Tensor,
    teacher_end: Tensor,
    final_teacher_clean: Tensor,
    start_logsnr: Tensor,
    end_logsnr: Tensor,
    timestep_indices: Tensor,
) -> Tensor:
    """Recover the one-step clean prediction implied by two teacher steps."""

    if not (
        noisy_start.shape == teacher_end.shape == final_teacher_clean.shape
    ):
        raise ValueError("progressive target tensors must share one shape")
    if timestep_indices.shape != (noisy_start.shape[0],):
        raise ValueError("timestep_indices must have shape [B]")
    alpha_start, _ = alpha_sigma(start_logsnr, noisy_start)
    alpha_end, _ = alpha_sigma(end_logsnr, noisy_start)
    noise_ratio = append_dims(
        torch.exp(
            0.5
            * (
                F.softplus(start_logsnr.float())
                - F.softplus(end_logsnr.float())
            )
        ),
        noisy_start.ndim,
    )
    denominator = alpha_end - noise_ratio * alpha_start
    safe_denominator = torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        denominator,
    )
    target = (teacher_end.float() - noise_ratio * noisy_start.float()) / safe_denominator
    first = append_dims(timestep_indices == 0, noisy_start.ndim)
    return torch.where(first, final_teacher_clean.float(), target)


def progressive_loss_per_sample(
    predicted_clean: Tensor,
    predicted_epsilon: Tensor,
    predicted_velocity: Tensor,
    target_clean: Tensor,
    target_epsilon: Tensor,
    target_velocity: Tensor,
    *,
    loss_weight: ProgressiveLossWeight,
) -> Tensor:
    shapes = {
        value.shape
        for value in (
            predicted_clean,
            predicted_epsilon,
            predicted_velocity,
            target_clean,
            target_epsilon,
            target_velocity,
        )
    }
    if len(shapes) != 1:
        raise ValueError("progressive predictions and targets must share one shape")

    def mse(left: Tensor, right: Tensor) -> Tensor:
        return (left.float() - right.float()).square().flatten(1).mean(dim=1)

    clean_mse = mse(predicted_clean, target_clean)
    epsilon_mse = mse(predicted_epsilon, target_epsilon)
    if loss_weight == "constant":
        return clean_mse
    if loss_weight == "snr":
        return epsilon_mse
    if loss_weight == "snr_trunc":
        return torch.maximum(clean_mse, epsilon_mse)
    if loss_weight == "v_mse":
        return mse(predicted_velocity, target_velocity)
    raise ValueError(f"unsupported progressive loss_weight: {loss_weight!r}")


__all__ = [
    "add_forward_noise",
    "alpha_sigma",
    "append_dims",
    "cosine_logsnr",
    "deterministic_ddim_step",
    "implied_clean_target",
    "prediction_to_clean_epsilon_velocity",
    "progressive_loss_per_sample",
]
