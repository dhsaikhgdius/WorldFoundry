"""Pure Flow-GRPO policy-objective math for stochastic transitions.

Key formulas:
  - Importance ratio: r = exp(log pi_new - log pi_old)
  - Clipped surrogate: L = -mean(min(r * A, clip(r, 1-eps, 1+eps) * A))
  - Approx KL: mean(log pi_old - log pi_new)

References:
  - Flow-GRPO: https://arxiv.org/abs/2505.05470
  - GRPO: https://arxiv.org/abs/2402.03300
  - PPO clip objective: https://arxiv.org/abs/1707.06347
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("post-training policy loss requires the 'train-core' extra") from error
    return torch


@dataclass(frozen=True, slots=True)
class ClippedPolicyLoss:
    loss: object
    per_transition: object
    ratio: object
    approx_kl: object
    clip_fraction: object
    lower_clip_fraction: object
    upper_clip_fraction: object


def clipped_policy_loss(
    new_log_probs: object,
    old_log_probs: object,
    advantages: object,
    *,
    clip_range: float,
    step_mask: object | None = None,
) -> ClippedPolicyLoss:
    """Return the PPO/GRPO clipped surrogate over ``[B,K]`` transitions."""

    torch = _require_torch()
    if not all(torch.is_tensor(value) for value in (new_log_probs, old_log_probs, advantages)):
        raise TypeError("new_log_probs, old_log_probs, and advantages must be torch.Tensor values")
    if new_log_probs.ndim != 2 or new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new_log_probs and old_log_probs must share shape [B,K]")
    if tuple(advantages.shape) == (int(new_log_probs.shape[0]),):
        expanded_advantages = advantages.reshape(-1, 1).expand_as(new_log_probs)
    elif advantages.shape == new_log_probs.shape:
        expanded_advantages = advantages
    else:
        raise ValueError("advantages must have shape [B] or [B,K]")
    resolved_clip = float(clip_range)
    if not isfinite(resolved_clip) or not 0 <= resolved_clip < 1:
        raise ValueError("clip_range must be finite and in [0,1)")

    new = new_log_probs.float()
    old = old_log_probs.detach().to(device=new.device, dtype=torch.float32)
    advantage = expanded_advantages.detach().to(device=new.device, dtype=torch.float32)
    if not bool(torch.isfinite(new).all() and torch.isfinite(old).all() and torch.isfinite(advantage).all()):
        raise FloatingPointError("policy objective inputs must be finite")
    log_ratio = new - old
    ratio = torch.exp(log_ratio)
    if not bool(torch.isfinite(ratio).all()):
        raise FloatingPointError("policy probability ratio overflowed")
    clipped_ratio = ratio.clamp(1.0 - resolved_clip, 1.0 + resolved_clip)
    per_transition = torch.maximum(-advantage * ratio, -advantage * clipped_ratio)

    if step_mask is None:
        weights = torch.ones_like(per_transition)
    else:
        if not torch.is_tensor(step_mask) or step_mask.shape != per_transition.shape:
            raise ValueError("step_mask must have shape [B,K]")
        weights = step_mask.to(device=new.device, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
            raise ValueError("step_mask must contain finite non-negative values")
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("step_mask selects no policy transitions")

    def weighted_mean(value: object) -> object:
        return (value * weights).sum() / denominator

    lower = ratio < 1.0 - resolved_clip
    upper = ratio > 1.0 + resolved_clip
    return ClippedPolicyLoss(
        loss=weighted_mean(per_transition),
        per_transition=per_transition,
        ratio=ratio,
        approx_kl=weighted_mean(0.5 * log_ratio.square()),
        clip_fraction=weighted_mean((lower | upper).float()),
        lower_clip_fraction=weighted_mean(lower.float()),
        upper_clip_fraction=weighted_mean(upper.float()),
    )


__all__ = ["ClippedPolicyLoss", "clipped_policy_loss"]
