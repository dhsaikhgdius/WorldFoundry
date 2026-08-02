"""ADD equations with explicit per-sample reductions.

Key formulas:
  - VP forward: x_t = alpha_t * x_0 + sigma_t * eps, alpha_t = sqrt(alpha_bar_t)
  - Distillation (Eq. 4): L = c(t) * ||x_student - x_teacher||^2
  - SDS weight: c(t) = alpha_t / (2 * sigma_t) * w_diffusion(t)
  - Generator hinge (Eq. 2): L_G = -sum_k D_k(fake)
  - Discriminator hinge (Eq. 3): L_D = sum_k [relu(1-D_k(real)) + relu(1+D_k(fake))]
  - R1 penalty: 0.5 * sum_k ||grad_{x_k} D_k(x_k)||^2

References:
  - Adversarial Diffusion Distillation (ADD): https://arxiv.org/abs/2311.17042
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import ADDNoiseSchedule
from .contracts import ADDDiscriminatorHeadOutput


def append_dims(value: Tensor, target_ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("value must be a tensor")
    if isinstance(target_ndim, bool) or not isinstance(target_ndim, int):
        raise TypeError("target_ndim must be an integer")
    if target_ndim < value.ndim:
        raise ValueError("target_ndim cannot be smaller than value.ndim")
    return value[(...,) + (None,) * (target_ndim - value.ndim)]


def schedule_coefficients(
    schedule: ADDNoiseSchedule,
    timesteps: Tensor,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gather the VP coefficients ``alpha_t`` and ``sigma_t``."""

    if not isinstance(schedule, ADDNoiseSchedule):
        raise TypeError("schedule must be ADDNoiseSchedule")
    if not isinstance(timesteps, Tensor) or timesteps.ndim != 1 or timesteps.dtype != torch.int64:
        raise TypeError("timesteps must be a one-dimensional int64 tensor")
    if not isinstance(reference, Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    if not reference.is_floating_point():
        raise TypeError("reference must be floating point")
    if timesteps.device != reference.device:
        raise ValueError("timesteps and reference must share a device")
    if timesteps.shape[0] != reference.shape[0]:
        raise ValueError("timesteps must match the reference batch size")
    if bool(((timesteps < 0) | (timesteps >= schedule.num_timesteps)).any()):
        raise ValueError("timesteps fall outside the noise schedule")
    alpha_power = torch.tensor(
        schedule.alpha_cumprods,
        device=reference.device,
        dtype=torch.float64,
    ).gather(0, timesteps)
    alpha = alpha_power.clamp_min(0.0).sqrt().to(dtype=torch.float32)
    sigma = (1.0 - alpha_power).clamp_min(0.0).sqrt().to(dtype=torch.float32)
    return alpha, sigma


def add_forward_noise(
    clean: Tensor,
    noise: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Apply ``x_t = alpha_t x_0 + sigma_t epsilon``."""

    if not isinstance(clean, Tensor) or clean.ndim < 2:
        raise TypeError("clean must be a [B,...] tensor")
    if not clean.is_floating_point() or not bool(torch.isfinite(clean).all()):
        raise ValueError("clean must be finite floating point")
    if not isinstance(noise, Tensor) or noise.shape != clean.shape:
        raise ValueError("noise must match clean")
    if noise.device != clean.device or noise.dtype != clean.dtype:
        raise ValueError("noise must share clean's device and dtype")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("noise must be finite")
    if alpha.shape != (clean.shape[0],) or sigma.shape != (clean.shape[0],):
        raise ValueError("alpha and sigma must have shape [B]")
    return (
        append_dims(alpha.to(device=clean.device, dtype=clean.dtype), clean.ndim) * clean
        + append_dims(sigma.to(device=clean.device, dtype=clean.dtype), clean.ndim) * noise
    )


def sample_student_timesteps(
    reference: Tensor,
    choices: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniformly sample the four ADD student timesteps per example."""

    if not isinstance(reference, Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    if len(choices) != 4:
        raise ValueError("ADD student timestep choices must contain four values")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in choices):
        raise ValueError("ADD student timestep choices must be non-negative integers")
    values = torch.tensor(choices, device=reference.device, dtype=torch.int64)
    indices = torch.randint(
        0,
        len(choices),
        (reference.shape[0],),
        device=reference.device,
        generator=generator,
    )
    return values[indices]


def sample_teacher_timesteps(
    reference: Tensor,
    *,
    minimum: int,
    maximum: int,
    probabilities: tuple[float, ...] | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the explicit teacher range uniformly or by configured mass."""

    if not isinstance(reference, Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or minimum < 0
        or maximum < minimum
    ):
        raise ValueError("teacher bounds must satisfy 0 <= minimum <= maximum")
    if probabilities is None:
        return torch.randint(
            minimum,
            maximum + 1,
            (reference.shape[0],),
            device=reference.device,
            generator=generator,
        )
    values = torch.as_tensor(probabilities, device=reference.device, dtype=torch.float64)
    if values.shape != (maximum - minimum + 1,):
        raise ValueError("teacher probabilities must align with the inclusive range")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or not bool(values.sum() > 0):
        raise ValueError("teacher probabilities must be finite, non-negative, and non-zero")
    indices = torch.multinomial(
        values,
        reference.shape[0],
        replacement=True,
        generator=generator,
    )
    return indices.to(dtype=torch.int64) + minimum


def distillation_weights(
    schedule: ADDNoiseSchedule,
    timesteps: Tensor,
    reference: Tensor,
    *,
    weighting: str,
) -> Tensor:
    """Return the report's exponential or SDS-equivalent ``c(t)``."""

    alpha, sigma = schedule_coefficients(schedule, timesteps, reference)
    if weighting == "exponential":
        if schedule.training_loss_weights is not None:
            raise ValueError("training_loss_weights are inert under exponential weighting")
        return alpha
    if weighting == "sds":
        if schedule.training_loss_weights is None:
            raise ValueError("SDS weighting requires teacher training_loss_weights")
        if bool((sigma <= 0).any()):
            raise ValueError("SDS weighting is singular at zero teacher noise")
        diffusion_weights = torch.tensor(
            schedule.training_loss_weights,
            device=reference.device,
            dtype=torch.float32,
        ).gather(0, timesteps)
        return alpha / (2.0 * sigma) * diffusion_weights
    raise ValueError(f"unsupported ADD distillation weighting: {weighting!r}")


def pixel_distillation_loss_per_sample(
    generated_images: Tensor,
    teacher_targets: Tensor,
    weights: Tensor,
) -> Tensor:
    """Weighted pixel-space squared distance from ADD Eq. (4)."""

    if not isinstance(generated_images, Tensor) or generated_images.ndim != 4:
        raise TypeError("generated_images must have shape [B,C,H,W]")
    if not isinstance(teacher_targets, Tensor) or teacher_targets.shape != generated_images.shape:
        raise ValueError("teacher_targets must match generated_images")
    if weights.shape != (generated_images.shape[0],):
        raise ValueError("distillation weights must have shape [B]")
    if (
        not generated_images.is_floating_point()
        or not teacher_targets.is_floating_point()
        or not weights.is_floating_point()
    ):
        raise TypeError("ADD pixel distillation inputs must be floating point")
    if not all(bool(torch.isfinite(value).all()) for value in (generated_images, teacher_targets, weights)):
        raise FloatingPointError("ADD pixel distillation inputs must be finite")
    squared_distance = (generated_images.float() - teacher_targets.float()).square()
    return squared_distance.flatten(1).mean(dim=1) * weights.float()


def reduce_head_logits(logits: Tensor) -> Tensor:
    """Reduce a scalar or patch discriminator head to one logit per sample."""

    if not isinstance(logits, Tensor) or logits.ndim < 1 or logits.shape[0] == 0:
        raise TypeError("head logits must be a non-empty tensor with a batch dimension")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("head logits must be finite")
    return logits.float().reshape(logits.shape[0], -1).mean(dim=1)


def generator_hinge_loss_per_sample(
    heads: Sequence[ADDDiscriminatorHeadOutput],
) -> Tensor:
    """ADD Eq. (2): negative sum of all feature-head scores."""

    if not heads:
        raise ValueError("ADD generator requires at least one discriminator head")
    reduced = [reduce_head_logits(head.logits) for head in heads]
    if any(value.shape != reduced[0].shape for value in reduced[1:]):
        raise ValueError("all discriminator heads must have the same batch size")
    return -torch.stack(reduced, dim=0).sum(dim=0)


def discriminator_hinge_loss_per_sample(
    real_heads: Sequence[ADDDiscriminatorHeadOutput],
    fake_heads: Sequence[ADDDiscriminatorHeadOutput],
) -> tuple[Tensor, Tensor, Tensor]:
    """ADD Eq. (3), preserving its sum over discriminator heads."""

    if not real_heads or len(real_heads) != len(fake_heads):
        raise ValueError("real and fake ADD head sequences must be aligned and non-empty")
    real_terms: list[Tensor] = []
    fake_terms: list[Tensor] = []
    for real, fake in zip(real_heads, fake_heads, strict=True):
        if (real.resolution, real.layer) != (fake.resolution, fake.layer):
            raise ValueError("real and fake discriminator head keys differ")
        real_logits = reduce_head_logits(real.logits)
        fake_logits = reduce_head_logits(fake.logits)
        if real_logits.shape != fake_logits.shape:
            raise ValueError("real and fake discriminator batches differ")
        real_terms.append(F.relu(1.0 - real_logits))
        fake_terms.append(F.relu(1.0 + fake_logits))
    real_loss = torch.stack(real_terms, dim=0).sum(dim=0)
    fake_loss = torch.stack(fake_terms, dim=0).sum(dim=0)
    return real_loss + fake_loss, real_loss, fake_loss


def feature_r1_penalty_per_sample(
    heads: Sequence[ADDDiscriminatorHeadOutput],
) -> Tensor:
    """R1/2 on each discriminator head input, then sum over heads."""

    if not heads:
        raise ValueError("R1 requires at least one real discriminator head")
    penalties: list[Tensor] = []
    for head in heads:
        features = head.features
        if not isinstance(features, Tensor) or features.ndim < 2:
            raise TypeError("R1 head features must be [B,...] tensors")
        if not features.requires_grad or not features.is_leaf:
            raise ValueError("R1 must be computed on detached leaf inputs to each head")
        logits = reduce_head_logits(head.logits)
        gradient = torch.autograd.grad(
            logits.sum(),
            features,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        penalties.append(0.5 * gradient.float().square().flatten(1).sum(dim=1))
    return torch.stack(penalties, dim=0).sum(dim=0)


__all__ = [
    "add_forward_noise",
    "append_dims",
    "discriminator_hinge_loss_per_sample",
    "distillation_weights",
    "feature_r1_penalty_per_sample",
    "generator_hinge_loss_per_sample",
    "pixel_distillation_loss_per_sample",
    "reduce_head_logits",
    "sample_student_timesteps",
    "sample_teacher_timesteps",
    "schedule_coefficients",
]
