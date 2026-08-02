"""Bagel Flow-UniGRPO policy and velocity-regularization math."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from ..flow_grpo.objective import clipped_policy_loss


@dataclass(frozen=True, slots=True)
class BagelFlowUniGRPOLoss:
    loss: torch.Tensor
    surrogate_loss: torch.Tensor
    velocity_mse: torch.Tensor
    per_transition: torch.Tensor
    ratio: torch.Tensor
    raw_ratio: torch.Tensor
    ratio_mean_bias: torch.Tensor | None
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    lower_clip_fraction: torch.Tensor
    upper_clip_fraction: torch.Tensor


def _transition_std(
    transition_scales: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    try:
        expanded = torch.broadcast_to(transition_scales.float(), target_shape)
    except RuntimeError as error:
        raise ValueError("transition_scales must broadcast to transition means") from error
    flattened = expanded.reshape(*target_shape[:2], -1)
    first = flattened[..., 0]
    if not torch.equal(flattened, first.unsqueeze(-1).expand_as(flattened)):
        raise ValueError("Bagel RatioNorm requires one scalar transition std per step")
    return first


def bagel_flow_unigrpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_transition_means: torch.Tensor,
    old_transition_means: torch.Tensor | None,
    transition_scales: torch.Tensor,
    sqrt_dt: torch.Tensor | None,
    advantages: torch.Tensor,
    policy_velocities: torch.Tensor,
    reference_velocities: torch.Tensor,
    *,
    clip_range: float,
    velocity_mse_weight: float,
    ratio_norm: bool,
    grad_reweight: bool,
) -> BagelFlowUniGRPOLoss:
    """Apply clipped flow RL plus a frozen-reference velocity MSE."""

    if new_log_probs.ndim != 2 or new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new_log_probs and old_log_probs must share shape [B,K]")
    if new_transition_means.ndim < 3 or new_transition_means.shape[:2] != new_log_probs.shape:
        raise ValueError("new_transition_means must start with [B,K]")
    if (
        policy_velocities.shape != new_transition_means.shape
        or reference_velocities.shape != new_transition_means.shape
    ):
        raise ValueError("policy/reference velocities must match transition mean shape")
    if tuple(advantages.shape) == (int(new_log_probs.shape[0]),):
        expanded_advantages = advantages.reshape(-1, 1).expand_as(new_log_probs)
    elif advantages.shape == new_log_probs.shape:
        expanded_advantages = advantages
    else:
        raise ValueError("advantages must have shape [B] or [B,K]")
    resolved_clip = float(clip_range)
    resolved_mse_weight = float(velocity_mse_weight)
    if not isfinite(resolved_clip) or not 0 < resolved_clip < 1:
        raise ValueError("clip_range must be finite and in (0,1)")
    if not isfinite(resolved_mse_weight) or resolved_mse_weight <= 0:
        raise ValueError("velocity_mse_weight must be finite and positive")
    if grad_reweight and not ratio_norm:
        raise ValueError("grad_reweight requires ratio_norm")

    new_logp = new_log_probs.float()
    old_logp = old_log_probs.detach().to(new_logp)
    advantage = expanded_advantages.detach().to(new_logp)
    policy_velocity = policy_velocities.float()
    reference_velocity = reference_velocities.detach().to(policy_velocity)
    tensors = (
        new_logp,
        old_logp,
        advantage,
        new_transition_means,
        policy_velocity,
        reference_velocity,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("Bagel Flow-UniGRPO inputs must be finite")

    raw_log_ratio = new_logp - old_logp
    raw_ratio = torch.exp(raw_log_ratio)
    ratio_mean_bias: torch.Tensor | None = None
    if ratio_norm:
        if old_transition_means is None or sqrt_dt is None:
            raise ValueError("Bagel RatioNorm requires frozen old means and replay sqrt_dt")
        if old_transition_means.shape != new_transition_means.shape:
            raise ValueError("old and new transition means must share shape")
        if sqrt_dt.shape != new_logp.shape:
            raise ValueError("sqrt_dt must have shape [B,K]")
        transition_std = _transition_std(
            transition_scales,
            tuple(new_transition_means.shape),
        ).detach()
        if not bool(torch.isfinite(transition_std).all()) or not bool((transition_std > 0).all()):
            raise ValueError("transition std must be finite and positive")
        latent_dims = tuple(range(2, new_transition_means.ndim))
        ratio_mean_bias = (
            new_transition_means.float() - old_transition_means.detach().to(new_transition_means).float()
        ).square().mean(dim=latent_dims) / (2.0 * transition_std.square())
        normalized_log_ratio = transition_std * (raw_log_ratio + ratio_mean_bias)
        ratio = torch.exp(normalized_log_ratio)
        clipped = ratio.clamp(1.0 - resolved_clip, 1.0 + resolved_clip)
        per_transition = torch.maximum(
            -advantage * ratio,
            -advantage * clipped,
        )
        if grad_reweight:
            step_sqrt = sqrt_dt.detach().to(new_logp).float()
            if not bool(torch.isfinite(step_sqrt).all()) or not bool((step_sqrt > 0).all()):
                raise ValueError("sqrt_dt must be finite and positive")
            inverse_dt = step_sqrt.square().reciprocal()
            weights = inverse_dt / inverse_dt.mean().clamp_min(1.0e-12)
            surrogate_loss = (per_transition * weights).mean()
        else:
            surrogate_loss = per_transition.mean()
        approx_kl = 0.5 * normalized_log_ratio.square().mean()
        lower = ratio < 1.0 - resolved_clip
        upper = ratio > 1.0 + resolved_clip
        clip_fraction = (lower | upper).float().mean()
        lower_clip_fraction = lower.float().mean()
        upper_clip_fraction = upper.float().mean()
    else:
        surrogate = clipped_policy_loss(
            new_logp,
            old_logp,
            advantage,
            clip_range=resolved_clip,
        )
        surrogate_loss = surrogate.loss
        per_transition = surrogate.per_transition
        ratio = surrogate.ratio
        approx_kl = surrogate.approx_kl
        clip_fraction = surrogate.clip_fraction
        lower_clip_fraction = surrogate.lower_clip_fraction
        upper_clip_fraction = surrogate.upper_clip_fraction

    velocity_mse = (policy_velocity - reference_velocity).square().mean()
    total = surrogate_loss + resolved_mse_weight * velocity_mse
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Bagel Flow-UniGRPO loss is non-finite")
    return BagelFlowUniGRPOLoss(
        loss=total,
        surrogate_loss=surrogate_loss,
        velocity_mse=velocity_mse,
        per_transition=per_transition,
        ratio=ratio,
        raw_ratio=raw_ratio,
        ratio_mean_bias=ratio_mean_bias,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        lower_clip_fraction=lower_clip_fraction,
        upper_clip_fraction=upper_clip_fraction,
    )


__all__ = ["BagelFlowUniGRPOLoss", "bagel_flow_unigrpo_loss"]
