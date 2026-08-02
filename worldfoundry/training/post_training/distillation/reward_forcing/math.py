"""Pure rewarded distribution-matching reduction math.

Key formulas:
  - Reward multiplier: w = exp(beta * MQ_normalized)
  - Re-DMD proxy: L = sum_i [0.5 * ||x_i - stop(x_i - g_i)||^2 * w_i * mask_i] / sum(mask)
    (reward weights scale numerator only, not the normalization denominator)

References:
  - Reward Forcing (Re-DMD): https://arxiv.org/abs/2512.04678
  - DMD: https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RewardedProxyLoss:
    loss: Tensor
    numerator: Tensor
    denominator: Tensor
    per_sample: Tensor


def reward_forcing_multiplier(rewards: Tensor, beta: float) -> Tensor:
    """Return the released ``exp(beta * normalized_MQ)`` weighting."""

    if not isinstance(rewards, Tensor) or rewards.ndim != 1:
        raise TypeError("Reward-Forcing rewards must be a one-dimensional tensor")
    if not rewards.is_floating_point():
        raise TypeError("Reward-Forcing rewards must use a floating dtype")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("Reward-Forcing rewards must be finite")
    resolved_beta = float(beta)
    if not isfinite(resolved_beta) or resolved_beta < 0:
        raise ValueError("Reward-Forcing beta must be finite and non-negative")
    multiplier = torch.exp(rewards.float() * resolved_beta)
    if not bool(torch.isfinite(multiplier).all()):
        raise FloatingPointError("Reward-Forcing reward multiplier overflowed")
    return multiplier


def _effective_weights(
    reference: Tensor,
    *,
    loss_mask: object | None,
    sample_weights: object | None,
) -> Tensor:
    batch_size = int(reference.shape[0])
    effective = torch.ones_like(reference, dtype=torch.float32)
    if loss_mask is not None:
        if not isinstance(loss_mask, Tensor):
            raise TypeError("loss_mask must be a torch.Tensor")
        mask = loss_mask
        if mask.ndim + 1 == reference.ndim and int(mask.shape[0]) == batch_size:
            mask = mask.unsqueeze(1)
        try:
            mask = torch.broadcast_to(mask, reference.shape)
        except RuntimeError as error:
            raise ValueError(
                f"loss_mask shape {tuple(mask.shape)} cannot broadcast to {tuple(reference.shape)}"
            ) from error
        mask = mask.to(device=reference.device, dtype=torch.float32)
        if not bool(torch.isfinite(mask).all()) or not bool((mask >= 0).all()):
            raise ValueError("loss_mask must be finite and non-negative")
        effective = effective * mask
    if sample_weights is not None:
        if not isinstance(sample_weights, Tensor) or tuple(sample_weights.shape) != (batch_size,):
            raise ValueError(f"sample_weights must have shape ({batch_size},)")
        weights = sample_weights.to(device=reference.device, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
            raise ValueError("sample_weights must be finite and non-negative")
        effective = effective * weights.reshape((batch_size,) + (1,) * (reference.ndim - 1))
    return effective


def rewarded_dmd_proxy_loss(
    generated_clean: Tensor,
    distribution_gradient: Tensor,
    reward_multiplier: Tensor,
    *,
    loss_mask: object | None = None,
    sample_weights: object | None = None,
) -> RewardedProxyLoss:
    """Apply per-sample reward weights without renormalizing them away.

    The double-precision proxy and leading one-half match the released Re-DMD
    implementation.  Base masks/sample weights determine the denominator;
    reward multipliers scale only the numerator.
    """

    if not isinstance(generated_clean, Tensor) or not isinstance(
        distribution_gradient,
        Tensor,
    ):
        raise TypeError("Reward-Forcing proxy inputs must be torch.Tensor values")
    if generated_clean.shape != distribution_gradient.shape or generated_clean.ndim < 2:
        raise ValueError("generated clean and DMD gradient must share a batched shape")
    if not generated_clean.is_floating_point() or not distribution_gradient.is_floating_point():
        raise TypeError("Reward-Forcing proxy inputs must use floating dtypes")
    if not isinstance(reward_multiplier, Tensor) or tuple(reward_multiplier.shape) != (int(generated_clean.shape[0]),):
        raise ValueError("reward_multiplier must have shape [B]")
    if not bool(torch.isfinite(reward_multiplier).all()) or not bool((reward_multiplier > 0).all()):
        raise ValueError("reward_multiplier must be finite and positive")

    generated = generated_clean.double()
    gradient = distribution_gradient.detach().double()
    target = (generated - gradient).detach()
    squared = 0.5 * (generated - target).square()
    effective = _effective_weights(
        squared,
        loss_mask=loss_mask,
        sample_weights=sample_weights,
    )
    flat_effective = effective.reshape(int(generated.shape[0]), -1)
    per_denominator = flat_effective.sum(dim=1)
    denominator = per_denominator.sum()
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("Reward-Forcing loss has no positive-weight elements")
    multiplier = reward_multiplier.detach().to(
        device=generated.device,
        dtype=torch.float64,
    )
    per_numerator = (squared.reshape(int(generated.shape[0]), -1) * flat_effective.double()).sum(dim=1) * multiplier
    numerator = per_numerator.sum()
    if not bool(torch.isfinite(numerator.detach())):
        raise FloatingPointError("Reward-Forcing proxy loss is non-finite")
    per_sample = torch.where(
        per_denominator > 0,
        per_numerator / per_denominator.double().clamp_min(1.0e-12),
        torch.zeros_like(per_numerator),
    )
    return RewardedProxyLoss(
        loss=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        per_sample=per_sample,
    )


__all__ = [
    "RewardedProxyLoss",
    "reward_forcing_multiplier",
    "rewarded_dmd_proxy_loss",
]
