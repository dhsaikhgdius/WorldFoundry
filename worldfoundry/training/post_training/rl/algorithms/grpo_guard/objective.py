"""GRPO-Guard policy objective with differentiable reverse-SDE mean bias."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("GRPO-Guard requires the 'train-core' extra") from error
    return torch


@dataclass(frozen=True, slots=True)
class GRPOGuardLoss:
    """Scalar objective and unreduced tensors used by the learner runtime."""

    loss: object
    per_transition: object
    ratio: object
    ratio_mean_bias: object
    scale: object
    sqrt_dt_mean: object
    ppo_kl: object
    approx_kl: object
    clip_fraction: object
    lower_clip_fraction: object
    upper_clip_fraction: object


def grpo_guard_policy_loss(
    new_log_probs: object,
    old_log_probs: object,
    new_transition_means: object,
    old_transition_means: object,
    std_dev_t: object,
    sqrt_dt: object,
    advantages: object,
    *,
    clip_range: float,
    advantage_clip_max: float,
    step_mask: object | None = None,
) -> GRPOGuardLoss:
    """Apply GRPO-Guard over batched stochastic transitions ``[B,K]``.

    The mean-drift bias is reduced only over latent dimensions, preserving one
    differentiable value per sample and transition.  Diffusion and timestep
    factors follow the shared-scalar reduction used by the objective.
    """

    torch = _require_torch()
    tensors = (
        new_log_probs,
        old_log_probs,
        new_transition_means,
        old_transition_means,
        std_dev_t,
        sqrt_dt,
        advantages,
    )
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("GRPO-Guard objective inputs must be torch.Tensor values")
    if new_log_probs.ndim != 2 or new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new_log_probs and old_log_probs must share shape [B,K]")
    if (
        new_transition_means.ndim < 3
        or new_transition_means.shape != old_transition_means.shape
        or new_transition_means.shape[:2] != new_log_probs.shape
    ):
        raise ValueError("transition means must share shape [B,K,...latent]")
    try:
        torch.broadcast_shapes(
            tuple(std_dev_t.shape),
            tuple(new_transition_means.shape),
        )
    except RuntimeError as error:
        raise ValueError("std_dev_t must broadcast to transition means") from error
    sqrt_shape = tuple(sqrt_dt.shape)
    if sqrt_shape != tuple(new_log_probs.shape):
        try:
            torch.broadcast_shapes(sqrt_shape, tuple(new_transition_means.shape))
        except RuntimeError as error:
            raise ValueError("sqrt_dt must have shape [B,K] or broadcast to transition means") from error
    if tuple(advantages.shape) == (int(new_log_probs.shape[0]),):
        expanded_advantages = advantages.reshape(-1, 1).expand_as(new_log_probs)
    elif advantages.shape == new_log_probs.shape:
        expanded_advantages = advantages
    else:
        raise ValueError("advantages must have shape [B] or [B,K]")

    resolved_clip = float(clip_range)
    resolved_advantage_clip = float(advantage_clip_max)
    if not isfinite(resolved_clip) or not 0 < resolved_clip < 1:
        raise ValueError("clip_range must be finite and in (0,1)")
    if not isfinite(resolved_advantage_clip) or resolved_advantage_clip <= 0:
        raise ValueError("advantage_clip_max must be finite and positive")

    new_logp = new_log_probs.float()
    old_logp = old_log_probs.detach().to(
        device=new_logp.device,
        dtype=torch.float32,
    )
    new_means = new_transition_means.float()
    old_means = old_transition_means.detach().to(
        device=new_means.device,
        dtype=torch.float32,
    )
    std = std_dev_t.detach().to(device=new_means.device, dtype=torch.float32)
    step_sqrt = sqrt_dt.detach().to(device=new_means.device, dtype=torch.float32)
    advantage = expanded_advantages.detach().to(
        device=new_logp.device,
        dtype=torch.float32,
    )
    if not bool(
        torch.isfinite(new_logp).all()
        and torch.isfinite(old_logp).all()
        and torch.isfinite(new_means).all()
        and torch.isfinite(old_means).all()
        and torch.isfinite(std).all()
        and torch.isfinite(step_sqrt).all()
        and torch.isfinite(advantage).all()
    ):
        raise FloatingPointError("GRPO-Guard objective inputs must be finite")
    if not bool((std > 0).all() and (step_sqrt > 0).all()):
        raise ValueError("std_dev_t and sqrt_dt must be positive")

    std_mean = std.mean()
    sqrt_dt_mean = step_sqrt.mean()
    scale = std_mean * sqrt_dt_mean
    latent_dims = tuple(range(2, new_means.ndim))
    mean_diff_sq = (new_means - old_means).square().mean(dim=latent_dims)
    ratio_mean_bias = mean_diff_sq / (2.0 * scale.square())
    log_ratio = new_logp - old_logp
    ratio = torch.exp((log_ratio + ratio_mean_bias) * scale)
    if not bool(torch.isfinite(ratio).all()):
        raise FloatingPointError("GRPO-Guard probability ratio overflowed")
    clipped_advantage = advantage.clamp(
        -resolved_advantage_clip,
        resolved_advantage_clip,
    )
    clipped_ratio = ratio.clamp(
        1.0 - resolved_clip,
        1.0 + resolved_clip,
    )
    per_transition = torch.maximum(
        -clipped_advantage * ratio,
        -clipped_advantage * clipped_ratio,
    )

    if step_mask is None:
        weights = torch.ones_like(per_transition)
    else:
        if not torch.is_tensor(step_mask) or step_mask.shape != per_transition.shape:
            raise ValueError("step_mask must have shape [B,K]")
        weights = step_mask.to(device=per_transition.device, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
            raise ValueError("step_mask must contain finite non-negative values")
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("step_mask selects no GRPO-Guard transitions")

    def weighted_mean(value: object) -> object:
        return (value * weights).sum() / denominator

    lower = ratio < 1.0 - resolved_clip
    upper = ratio > 1.0 + resolved_clip
    return GRPOGuardLoss(
        loss=weighted_mean(per_transition) / sqrt_dt_mean.square(),
        per_transition=per_transition,
        ratio=ratio,
        ratio_mean_bias=ratio_mean_bias,
        scale=scale,
        sqrt_dt_mean=sqrt_dt_mean,
        ppo_kl=weighted_mean(-log_ratio),
        approx_kl=weighted_mean(0.5 * log_ratio.square()),
        clip_fraction=weighted_mean((lower | upper).float()),
        lower_clip_fraction=weighted_mean(lower.float()),
        upper_clip_fraction=weighted_mean(upper.float()),
    )


__all__ = ["GRPOGuardLoss", "grpo_guard_policy_loss"]
