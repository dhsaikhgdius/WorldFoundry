"""Pure formula components for scale-wise flow distillation.

Key formulas:
  - Flow noise: x_t = sigma * eps + (1 - sigma) * x_0
  - Clean from velocity: x_0 = x_t - sigma * v
  - DMD proxy: g = (x_gen-x_real - (x_gen-x_fake)) / norm; L = 0.5 * ||x_gen - stop(x_gen-g)||^2
  - Logistic GAN: L_D = softplus(D(fake)) + softplus(-D(real)); L_G = softplus(-D(fake))
  - MMD (RBF): MMD^2 = E[k(real,real)] + E[k(fake,fake)] - 2E[k(real,fake)]

References:
  - Scale-wise Distillation (SwD): https://arxiv.org/abs/2503.16397
  - DMD: https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

from math import isfinite

import torch
import torch.nn.functional as F


def _matching_latents(*values: torch.Tensor) -> None:
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("scale-wise latent inputs must be torch.Tensor values")
    if values[0].ndim < 2 or any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("scale-wise latent inputs must have one matching [B,...] shape")


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


def flow_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    _matching_latents(clean, noise)
    sigma = expand_sigmas(sigmas, clean).to(device=clean.device, dtype=clean.dtype)
    return sigma * noise + (1.0 - sigma) * clean


def clean_from_velocity(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    _matching_latents(noisy, velocity)
    sigma = expand_sigmas(sigmas, noisy).to(device=noisy.device, dtype=noisy.dtype)
    return noisy - sigma * velocity


def classifier_free_guidance(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    _matching_latents(unconditional, conditional)
    scale = float(guidance_scale)
    if not isfinite(scale) or scale < 0.0:
        raise ValueError("guidance_scale must be finite and non-negative")
    return unconditional + scale * (conditional - unconditional)


def upscale_previous_latents(
    previous_latents: torch.Tensor,
    *,
    current_scale: int,
) -> torch.Tensor:
    if not isinstance(previous_latents, torch.Tensor) or previous_latents.ndim != 4:
        raise TypeError("previous_latents must be a [B,C,H,W] torch.Tensor")
    if isinstance(current_scale, bool) or int(current_scale) <= 0:
        raise ValueError("current_scale must be positive")
    target = int(current_scale)
    if previous_latents.shape[-2:] == (target, target):
        return previous_latents
    return F.interpolate(
        previous_latents,
        size=(target, target),
        mode="bicubic",
        align_corners=False,
    )


def dmd_loss_per_sample(
    generated: torch.Tensor,
    real_clean: torch.Tensor,
    fake_clean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Released SwD DMD pseudo-target and per-sample objective."""

    _matching_latents(generated, real_clean, fake_clean)
    axes = tuple(range(1, generated.ndim))
    with torch.no_grad():
        p_real = generated.float() - real_clean.float()
        p_fake = generated.float() - fake_clean.float()
        normalizer = p_real.abs().mean(dim=axes, keepdim=True)
        gradient = torch.nan_to_num((p_real - p_fake) / normalizer)
        target = generated.float() - gradient
    per_sample = 0.5 * (generated.float() - target).square().flatten(1).mean(1)
    return per_sample, gradient, normalizer


def fake_diffusion_loss_per_sample(
    fake_clean_prediction: torch.Tensor,
    generated: torch.Tensor,
) -> torch.Tensor:
    _matching_latents(fake_clean_prediction, generated)
    return (
        fake_clean_prediction.float() - generated.detach().float()
    ).square().flatten(1).mean(1)


def _mean_logits(logits: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not logits or not all(isinstance(value, torch.Tensor) for value in logits):
        raise TypeError("classifier logits must be a non-empty tensor tuple")
    if any(value.shape != logits[0].shape for value in logits[1:]):
        raise ValueError("classifier logits must have matching shapes")
    return torch.stack(tuple(value.float() for value in logits), dim=0).mean(dim=0)


def discriminator_logistic_loss(
    fake_logits: tuple[torch.Tensor, ...],
    real_logits: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    fake = _mean_logits(fake_logits)
    real = _mean_logits(real_logits)
    if fake.shape != real.shape:
        raise ValueError("real and fake discriminator logits must have matching shapes")
    return F.softplus(fake).reshape(fake.shape[0], -1).mean(1) + F.softplus(
        -real
    ).reshape(real.shape[0], -1).mean(1)


def generator_logistic_loss(
    fake_logits: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    fake = _mean_logits(fake_logits)
    return F.softplus(-fake).reshape(fake.shape[0], -1).mean(1)


def pool_token_features(
    features: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if not features or not all(isinstance(value, torch.Tensor) for value in features):
        raise TypeError("features must be a non-empty tensor tuple")
    if any(value.ndim < 3 for value in features):
        raise ValueError("features must have shape [B,N,D...] for token pooling")
    return tuple(value.float().mean(dim=1) for value in features)


def mmd_loss(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    *,
    kernel: str,
    rbf_sigma: float,
    batch_mmd: bool,
    huber_c: float,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Released token-feature MMD, including its pseudo-Huber linear kernel."""

    if not isinstance(real_features, torch.Tensor) or not isinstance(
        fake_features,
        torch.Tensor,
    ):
        raise TypeError("MMD features must be torch.Tensor values")
    if real_features.ndim != 3 or fake_features.shape != real_features.shape:
        raise ValueError("MMD features must have one matching [B,N,D] shape")
    resolved_kernel = str(kernel).strip().lower()
    sigma = float(rbf_sigma)
    c = float(huber_c)
    eps = float(epsilon)
    if not isfinite(sigma) or sigma <= 0.0:
        raise ValueError("rbf_sigma must be finite and positive")
    if not isfinite(c) or c < 0.0 or not isfinite(eps) or eps <= 0.0:
        raise ValueError("huber_c and epsilon are invalid")
    real = real_features.float()
    fake = fake_features.float()
    if batch_mmd:
        real = real.flatten(0, 1).unsqueeze(0)
        fake = fake.flatten(0, 1).unsqueeze(0)
    if resolved_kernel == "linear":
        squared = (real.mean(dim=1) - fake.mean(dim=1)).square()
        return ((squared + c**2).sqrt().clamp_min(eps) - c).mean()
    if resolved_kernel != "rbf":
        raise ValueError("kernel must be 'linear' or 'rbf'")
    tokens = real.shape[1]
    if tokens < 2:
        raise ValueError("unbiased RBF MMD requires at least two tokens")
    real_real = torch.bmm(real, real.transpose(1, 2))
    fake_fake = torch.bmm(fake, fake.transpose(1, 2))
    real_fake = torch.bmm(real, fake.transpose(1, 2))
    real_norm = torch.diagonal(real_real, dim1=1, dim2=2).unsqueeze(1)
    fake_norm = torch.diagonal(fake_fake, dim1=1, dim2=2).unsqueeze(1)
    distance_real = real_norm.transpose(1, 2) + real_norm - 2.0 * real_real
    distance_fake = fake_norm.transpose(1, 2) + fake_norm - 2.0 * fake_fake
    distance_cross = real_norm.transpose(1, 2) + fake_norm - 2.0 * real_fake
    alpha = 1.0 / (2.0 * sigma**2)
    kernel_real = torch.exp(-alpha * distance_real)
    kernel_fake = torch.exp(-alpha * distance_fake)
    kernel_cross = torch.exp(-alpha * distance_cross)
    denominator = tokens * (tokens - 1)
    real_mean = (kernel_real.sum(dim=(1, 2)) - tokens) / denominator
    fake_mean = (kernel_fake.sum(dim=(1, 2)) - tokens) / denominator
    cross_mean = kernel_cross.sum(dim=(1, 2)) / (tokens * tokens)
    return (real_mean + fake_mean - 2.0 * cross_mean).mean()


__all__ = [
    "classifier_free_guidance",
    "clean_from_velocity",
    "discriminator_logistic_loss",
    "dmd_loss_per_sample",
    "expand_sigmas",
    "fake_diffusion_loss_per_sample",
    "flow_noise",
    "generator_logistic_loss",
    "mmd_loss",
    "pool_token_features",
    "upscale_previous_latents",
]
