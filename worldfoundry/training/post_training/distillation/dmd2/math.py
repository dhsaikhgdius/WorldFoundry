"""Formula-level DMD2 losses with explicit per-sample reductions.

Key formulas:
  - DMD gradient: g = (x0_fake - x0_real) / mean|x - x0_real|
  - DMD proxy: L = 0.5 * ||x_gen - stop(x_gen - g)||^2  (grad w.r.t. x_gen equals g)
  - Generator GAN: L_G = softplus(-D(fake))
  - Discriminator GAN: L_D = softplus(D(fake)) + softplus(-D(real))

References:
  - DMD2: https://arxiv.org/abs/2405.14867
  - DMD: https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("native DMD2 requires the 'train-core' extra") from error
    return torch


def _per_sample_logits(logits: object, *, field_name: str) -> object:
    torch = _require_torch()
    if not torch.is_tensor(logits) or logits.ndim < 1:
        raise TypeError(f"{field_name} must be a tensor with a batch dimension")
    flattened = logits.float().reshape(int(logits.shape[0]), -1)
    if flattened.shape[1] != 1:
        raise ValueError(f"{field_name} must contain exactly one logit per sample")
    if not bool(torch.isfinite(flattened).all()):
        raise ValueError(f"{field_name} must be finite")
    return flattened[:, 0]


def dmd2_distribution_gradient(
    score_sample: object,
    fake_score_clean: object,
    real_score_clean: object,
    *,
    normalization_axes: tuple[int, ...],
    normalization_epsilon: float = 0.0,
    calculation_dtype: str = "float32",
) -> tuple[object, object]:
    """Return ``(x0_fake-x0_real)/mean(|x-x0_real|)`` per sample."""

    torch = _require_torch()
    values = (score_sample, fake_score_clean, real_score_clean)
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("DMD2 distribution-gradient inputs must be tensors")
    if score_sample.ndim < 2 or any(value.shape != score_sample.shape for value in values[1:]):
        raise ValueError("DMD2 distribution-gradient inputs must share a [B,...] shape")
    axes = tuple(int(axis) for axis in normalization_axes)
    expected_axes = tuple(range(1, score_sample.ndim))
    if axes != expected_axes:
        raise ValueError(
            "normalization_axes must explicitly include every non-batch latent axis; "
            f"expected {expected_axes}, got {axes}"
        )
    epsilon = float(normalization_epsilon)
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("normalization_epsilon must be finite and non-negative")
    dtype = torch.float64 if calculation_dtype == "float64" else torch.float32
    if calculation_dtype not in {"float32", "float64"}:
        raise ValueError("calculation_dtype must be 'float32' or 'float64'")
    sample = score_sample.to(dtype=dtype)
    fake = fake_score_clean.to(dtype=dtype)
    real = real_score_clean.to(dtype=dtype)
    denominator = (sample - real).abs().mean(dim=axes, keepdim=True)
    divisor = denominator.clamp_min(epsilon) if epsilon > 0 else denominator
    gradient = torch.nan_to_num((fake - real) / divisor)
    return gradient, denominator.reshape(int(sample.shape[0]))


def dmd2_proxy_loss_per_sample(
    generated_clean: object,
    distribution_gradient: object,
    *,
    calculation_dtype: str = "float32",
) -> object:
    """Return the half-MSE proxy whose gradient equals the supplied DM field."""

    torch = _require_torch()
    if not torch.is_tensor(generated_clean) or not torch.is_tensor(distribution_gradient):
        raise TypeError("DMD2 proxy-loss inputs must be tensors")
    if generated_clean.ndim < 2 or generated_clean.shape != distribution_gradient.shape:
        raise ValueError("DMD2 proxy-loss inputs must share a [B,...] shape")
    dtype = torch.float64 if calculation_dtype == "float64" else torch.float32
    if calculation_dtype not in {"float32", "float64"}:
        raise ValueError("calculation_dtype must be 'float32' or 'float64'")
    generated = generated_clean.to(dtype=dtype)
    target = (generated - distribution_gradient.to(dtype=dtype)).detach()
    return 0.5 * (generated - target).square().reshape(int(generated.shape[0]), -1).mean(dim=1)


def dmd2_generator_adversarial_loss(fake_logits: object) -> object:
    """Non-saturating generator loss ``softplus(-D(fake))`` per sample."""

    torch = _require_torch()
    fake = _per_sample_logits(fake_logits, field_name="fake_logits")
    return torch.nn.functional.softplus(-fake)


def dmd2_guidance_adversarial_loss(real_logits: object, fake_logits: object) -> object:
    """Discriminator loss ``softplus(D(fake))+softplus(-D(real))`` per sample."""

    torch = _require_torch()
    real = _per_sample_logits(real_logits, field_name="real_logits")
    fake = _per_sample_logits(fake_logits, field_name="fake_logits")
    if real.shape != fake.shape:
        raise ValueError("real_logits and fake_logits must have equal batch size")
    return torch.nn.functional.softplus(fake) + torch.nn.functional.softplus(-real)


def dmd2_weighted_total(
    components: Mapping[str, object],
    weights: Mapping[str, float],
) -> object:
    """Combine aligned per-sample components while consuming every declared weight."""

    torch = _require_torch()
    if set(components) != set(weights) or not components:
        raise ValueError("DMD2 loss components and weights must have identical non-empty keys")
    result = None
    expected_shape = None
    for name, component in components.items():
        if not torch.is_tensor(component) or component.ndim != 1:
            raise TypeError(f"{name} must be a per-sample rank-one tensor")
        if expected_shape is None:
            expected_shape = component.shape
        elif component.shape != expected_shape:
            raise ValueError("DMD2 per-sample loss components must have equal shapes")
        weight = float(weights[name])
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"{name} weight must be finite and non-negative")
        term = component.float() * weight
        result = term if result is None else result + term
    assert result is not None
    return result


__all__ = [
    "dmd2_distribution_gradient",
    "dmd2_generator_adversarial_loss",
    "dmd2_guidance_adversarial_loss",
    "dmd2_proxy_loss_per_sample",
    "dmd2_weighted_total",
]
