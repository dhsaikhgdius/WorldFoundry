"""Source-equivalent SiD-DiT scalar formulas with per-sample outputs.

Key formulas:
  - Score weight schemes: w ~ 1/|x_gen-x_teacher|, 1/(sigma/(1-sigma)), 1/sigma^2, (1-sigma)^2, ...
  - Generator (score identity): L = sum w * (x_teacher - x_fake) * (x_fake - x_gen)  [alpha=1]
    or w * (x_teacher - x_fake) * (x_teacher - x_gen - alpha*(x_teacher - x_fake))
  - Fake-score flow: L = 0.5 * ||v_pred - v_target||^2
  - Adversarial BCE (Diffusion GAN): L_G = BCE(D(fake), 1) * w * |latent|; L_D = 0.5*(BCE(D(real),1)+BCE(D(fake),0))

References:
  - SiD (one-step): https://arxiv.org/abs/2404.04057
  - Few-step SiD: https://arxiv.org/abs/2505.12674
"""

from __future__ import annotations

from math import isfinite


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("native SiD requires the 'train-core' extra") from error
    return torch


def _per_sample_sum(value: object) -> object:
    return value.float().reshape(value.shape[0], -1).sum(dim=1)


def sid_classifier_free_guidance(
    unconditional: object,
    conditional: object,
    guidance_scale: float,
) -> object:
    """Return ``uncond + scale * (cond - uncond)``."""

    if getattr(unconditional, "shape", None) != getattr(conditional, "shape", None):
        raise ValueError("SiD conditional and unconditional predictions must match")
    scale = float(guidance_scale)
    if not isfinite(scale):
        raise ValueError("guidance_scale must be finite")
    return unconditional + scale * (conditional - unconditional)


def sid_score_weight(
    sigmas: object,
    *,
    scheme: str,
    epsilon: float,
    generated: object | None = None,
    teacher_clean: object | None = None,
) -> object:
    """Compute the official SiD weighting factor as one value per sample."""

    torch = _require_torch()
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise TypeError("sigmas must be a one-dimensional floating tensor")
    if not bool(torch.isfinite(sigmas).all()) or not bool(((0 <= sigmas) & (sigmas <= 1)).all()):
        raise ValueError("sigmas must be finite values in [0,1]")
    floor = float(epsilon)
    if not isfinite(floor) or floor <= 0:
        raise ValueError("epsilon must be finite and positive")
    normalized = str(scheme).strip().lower().replace("_", "-")
    sigma = sigmas.float()
    if normalized == "sid-legacy":
        if (
            not torch.is_tensor(generated)
            or not torch.is_tensor(teacher_clean)
            or generated.shape != teacher_clean.shape
            or int(generated.shape[0]) != int(sigmas.shape[0])
        ):
            raise ValueError("sid-legacy weighting requires aligned generated and teacher tensors")
        return (generated.float() - teacher_clean.float()).abs().reshape(
            generated.shape[0], -1
        ).mean(dim=1).clamp_min(floor).reciprocal()
    if normalized == "snr-sqrt":
        return (sigma / (1.0 - sigma)).clamp_min(floor).reciprocal()
    if normalized == "snr":
        return (sigma / (1.0 - sigma)).square().clamp_min(floor).reciprocal()
    if normalized == "1-over-sigma2":
        return sigma.square().clamp_min(floor).reciprocal()
    if normalized == "1-over-sigma":
        return sigma.clamp_min(floor).reciprocal()
    if normalized == "1-minus-sigma-squared":
        return (1.0 - sigma).square()
    if normalized == "1-minus-sigma":
        return 1.0 - sigma
    raise ValueError(f"unsupported SiD score weighting: {scheme!r}")


def sid_generator_loss_per_sample(
    generated: object,
    teacher_clean: object,
    fake_clean: object,
    score_weight: object,
    *,
    alpha: float,
) -> object:
    """General score-identity generator formula, summed over latent elements."""

    torch = _require_torch()
    if not all(torch.is_tensor(value) for value in (generated, teacher_clean, fake_clean, score_weight)):
        raise TypeError("SiD generator inputs must be tensors")
    if generated.shape != teacher_clean.shape or generated.shape != fake_clean.shape:
        raise ValueError("SiD generator prediction tensors must share shape")
    if score_weight.shape != (generated.shape[0],):
        raise ValueError("score_weight must have shape [B]")
    resolved_alpha = float(alpha)
    if not isfinite(resolved_alpha):
        raise ValueError("alpha must be finite")
    difference = teacher_clean.float() - fake_clean.float()
    if resolved_alpha == 1.0:
        factors = difference * (fake_clean.float() - generated.float())
    else:
        factors = difference * (
            teacher_clean.float()
            - generated.float()
            - resolved_alpha * difference
        )
    weight = score_weight.float().reshape(
        (score_weight.shape[0],) + (1,) * (generated.ndim - 1)
    )
    return _per_sample_sum(factors * weight)


def sid_fake_score_flow_loss_per_sample(prediction: object, target: object) -> object:
    torch = _require_torch()
    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("SiD fake-score flow inputs must be tensors")
    if prediction.shape != target.shape:
        raise ValueError("SiD fake-score flow prediction and target must match")
    return _per_sample_sum((prediction.float() - target.float()).square())


def sid_generator_adversarial_loss_per_sample(
    fake_logits: object,
    score_weight: object,
    *,
    latent_elements: int,
) -> object:
    """Official clamped BCE generator loss, scaled to latent element count."""

    torch = _require_torch()
    if not torch.is_tensor(fake_logits) or not torch.is_tensor(score_weight):
        raise TypeError("SiD generator GAN inputs must be tensors")
    if fake_logits.ndim < 1 or score_weight.shape != (fake_logits.shape[0],):
        raise ValueError("SiD generator GAN inputs must share batch size")
    if isinstance(latent_elements, bool) or int(latent_elements) <= 0:
        raise ValueError("latent_elements must be positive")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_logits.float().clamp(-10.0, 10.0),
        torch.ones_like(fake_logits, dtype=torch.float32),
        reduction="none",
    ).reshape(fake_logits.shape[0], -1).mean(dim=1)
    return bce * score_weight.float() * int(latent_elements)


def sid_fake_score_adversarial_loss_per_sample(
    real_logits: object,
    fake_logits: object,
    *,
    latent_elements: int,
) -> object:
    """Official average of real/fake clamped BCE discriminator terms."""

    torch = _require_torch()
    if not torch.is_tensor(real_logits) or not torch.is_tensor(fake_logits):
        raise TypeError("SiD fake-score GAN inputs must be tensors")
    if real_logits.ndim < 1 or fake_logits.ndim < 1 or real_logits.shape[0] != fake_logits.shape[0]:
        raise ValueError("SiD fake-score GAN logits must share batch size")
    if isinstance(latent_elements, bool) or int(latent_elements) <= 0:
        raise ValueError("latent_elements must be positive")
    real = torch.nn.functional.binary_cross_entropy_with_logits(
        real_logits.float().clamp(-10.0, 10.0),
        torch.ones_like(real_logits, dtype=torch.float32),
        reduction="none",
    ).reshape(real_logits.shape[0], -1).mean(dim=1)
    fake = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_logits.float().clamp(-10.0, 10.0),
        torch.zeros_like(fake_logits, dtype=torch.float32),
        reduction="none",
    ).reshape(fake_logits.shape[0], -1).mean(dim=1)
    return 0.5 * (real + fake) * int(latent_elements)


__all__ = [
    "sid_classifier_free_guidance",
    "sid_fake_score_adversarial_loss_per_sample",
    "sid_fake_score_flow_loss_per_sample",
    "sid_generator_adversarial_loss_per_sample",
    "sid_generator_loss_per_sample",
    "sid_score_weight",
]
