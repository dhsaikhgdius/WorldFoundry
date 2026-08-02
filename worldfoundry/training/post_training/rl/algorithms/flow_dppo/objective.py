"""KL-advantage masking objective for flow-policy optimization.

Key formulas:
  - Per-transition KL: KL_old = mean ||mu_new - mu_old||^2 / (2 * sigma^2)
  - Keep mask: keep = (KL_old <= threshold) OR (A >= 0)
  - Clipped policy loss on masked transitions (same as PPO/GRPO clip)

References:
  - Flow-DPPO (KL-masked policy optimization): https://arxiv.org/abs/2505.05470
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Flow-DPPO requires the 'train-core' extra") from error
    return torch


@dataclass(frozen=True, slots=True)
class FlowDPPOLoss:
    """Scalar Flow-DPPO objective plus unreduced audit tensors."""

    loss: object
    per_transition: object
    ratio: object
    old_policy_kl: object
    keep_mask: object
    masked_fraction: object
    positive_masked_fraction: object
    negative_masked_fraction: object
    approx_kl: object


def flow_dppo_policy_loss(
    new_log_probs: object,
    old_log_probs: object,
    new_transition_means: object,
    old_transition_means: object,
    transition_scales: object,
    advantages: object,
    *,
    kl_mask_threshold: float = 1.0e-5,
    add_kl_coefficient: bool = True,
    step_mask: object | None = None,
) -> FlowDPPOLoss:
    """Apply Flow-DPPO's equal-covariance KL-ADV masking over ``[B,K]``.

    A high-KL transition is removed only when its importance-ratio direction
    already amplifies the reward-improving move.  ``torch.where`` is used for
    the mask so an overflowed ratio that is removed cannot form ``inf * 0``.
    """

    torch = _require_torch()
    tensors = (
        new_log_probs,
        old_log_probs,
        new_transition_means,
        old_transition_means,
        transition_scales,
        advantages,
    )
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("Flow-DPPO objective inputs must be torch.Tensor values")
    if new_log_probs.ndim != 2 or new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new_log_probs and old_log_probs must share shape [B,K]")
    if (
        new_transition_means.ndim < 3
        or new_transition_means.shape != old_transition_means.shape
        or new_transition_means.shape[:2] != new_log_probs.shape
    ):
        raise ValueError("transition means must share shape [B,K,...latent]")
    try:
        torch.broadcast_shapes(tuple(transition_scales.shape), tuple(new_transition_means.shape))
    except RuntimeError as error:
        raise ValueError("transition_scales must broadcast to transition means") from error
    if tuple(advantages.shape) == (int(new_log_probs.shape[0]),):
        expanded_advantages = advantages.reshape(-1, 1).expand_as(new_log_probs)
    elif advantages.shape == new_log_probs.shape:
        expanded_advantages = advantages
    else:
        raise ValueError("advantages must have shape [B] or [B,K]")
    threshold = float(kl_mask_threshold)
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("kl_mask_threshold must be finite and non-negative")
    if not isinstance(add_kl_coefficient, bool):
        raise TypeError("add_kl_coefficient must be a bool")

    new_logp = new_log_probs.float()
    old_logp = old_log_probs.detach().to(device=new_logp.device, dtype=torch.float32)
    new_means = new_transition_means.float()
    old_means = old_transition_means.detach().to(device=new_means.device, dtype=torch.float32)
    advantage = expanded_advantages.detach().to(device=new_logp.device, dtype=torch.float32)
    scales = transition_scales.detach().to(device=new_means.device, dtype=torch.float32)
    if not bool(
        torch.isfinite(new_logp).all()
        and torch.isfinite(old_logp).all()
        and torch.isfinite(new_means).all()
        and torch.isfinite(old_means).all()
        and torch.isfinite(advantage).all()
        and torch.isfinite(scales).all()
    ):
        raise FloatingPointError("Flow-DPPO objective inputs must be finite")
    if add_kl_coefficient and not bool((scales > 0).all()):
        raise ValueError("likelihood-bearing transition scales must be positive")

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    denominator = 2.0 * scales.square() if add_kl_coefficient else 2.0
    kl_elements = (new_means - old_means).square() / denominator
    latent_dims = tuple(range(2, kl_elements.ndim))
    old_policy_kl = kl_elements.mean(dim=latent_dims)
    high_kl = old_policy_kl >= threshold
    positive_remove = high_kl & (ratio > 1.0) & (advantage > 0)
    negative_remove = high_kl & (ratio < 1.0) & (advantage < 0)
    remove = positive_remove | negative_remove
    keep = (~remove).detach()
    zero = torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    per_transition = torch.where(keep, -advantage * ratio, zero)

    if step_mask is None:
        weights = torch.ones_like(per_transition)
    else:
        if not torch.is_tensor(step_mask) or step_mask.shape != per_transition.shape:
            raise ValueError("step_mask must have shape [B,K]")
        weights = step_mask.to(device=per_transition.device, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
            raise ValueError("step_mask must contain finite non-negative values")
    weight_sum = weights.sum()
    if not bool(weight_sum > 0):
        raise ValueError("step_mask selects no Flow-DPPO transitions")

    def weighted_mean(value: object) -> object:
        return (value * weights).sum() / weight_sum

    return FlowDPPOLoss(
        loss=weighted_mean(per_transition),
        per_transition=per_transition,
        ratio=ratio,
        old_policy_kl=old_policy_kl,
        keep_mask=keep,
        masked_fraction=weighted_mean(remove.float()),
        positive_masked_fraction=weighted_mean(positive_remove.float()),
        negative_masked_fraction=weighted_mean(negative_remove.float()),
        approx_kl=weighted_mean(0.5 * log_ratio.square()),
    )


__all__ = ["FlowDPPOLoss", "flow_dppo_policy_loss"]
