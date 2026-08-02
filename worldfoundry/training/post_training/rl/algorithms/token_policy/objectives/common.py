"""Shared tensors and validation for packed token-policy objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True, slots=True)
class TokenObjective:
    """Unreduced policy losses and diagnostics in token or sequence space."""

    losses: torch.Tensor
    ratio: torch.Tensor
    log_ratio: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    keep_mask: torch.Tensor | None = None
    regularizer: torch.Tensor | None = None


def validated_policy_inputs(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate aligned vectors and freeze old-policy and advantage inputs."""

    values = (new_log_probs, old_log_probs, advantages)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("policy objective inputs must be torch.Tensor values")
    if new_log_probs.ndim != 1 or old_log_probs.shape != new_log_probs.shape or advantages.shape != new_log_probs.shape:
        raise ValueError("new_log_probs, old_log_probs, and advantages must share shape [N]")
    if not new_log_probs.is_floating_point():
        raise TypeError("new_log_probs must be floating point")
    new_logp = new_log_probs
    old_logp = old_log_probs.detach().to(
        device=new_logp.device,
        dtype=new_logp.dtype,
    )
    advantage = advantages.detach().to(
        device=new_logp.device,
        dtype=new_logp.dtype,
    )
    if not bool(torch.isfinite(new_logp).all() and torch.isfinite(old_logp).all() and torch.isfinite(advantage).all()):
        raise FloatingPointError("policy objective inputs must be finite")
    return new_logp, old_logp, advantage


def validated_positive_float(value: float, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def clipped_policy_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
    clip_range_high: float | None = None,
) -> TokenObjective:
    """PPO clipped surrogate over any aligned one-dimensional policy units."""

    new_logp, old_logp, advantage = validated_policy_inputs(
        new_log_probs,
        old_log_probs,
        advantages,
    )
    lower = float(clip_range)
    if not isfinite(lower) or not 0 <= lower < 1:
        raise ValueError("clip_range must be finite and in [0,1)")
    upper = lower if clip_range_high is None else float(clip_range_high)
    if not isfinite(upper) or upper < 0:
        raise ValueError("clip_range_high must be finite and non-negative")
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    if not bool(torch.isfinite(ratio).all()):
        raise FloatingPointError("policy ratio overflowed")
    clipped_ratio = ratio.clamp(1.0 - lower, 1.0 + upper)
    losses = torch.maximum(-advantage * ratio, -advantage * clipped_ratio)
    above = ratio - 1.0 > upper
    below = 1.0 - ratio > lower
    metrics = {
        "approx_kl": 0.5 * log_ratio.square().mean(),
        "clip_fraction": (above | below).float().mean(),
        "clipfrac_gt_one": above.float().mean(),
        "clipfrac_lt_one": below.float().mean(),
    }
    return TokenObjective(
        losses=losses,
        ratio=ratio,
        log_ratio=log_ratio,
        metrics=metrics,
    )


__all__ = [
    "TokenObjective",
    "clipped_policy_objective",
    "validated_policy_inputs",
    "validated_positive_float",
]
