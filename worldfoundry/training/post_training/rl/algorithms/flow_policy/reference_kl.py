"""Reference-policy KL shared by flow-policy learner algorithms.

Key formulas:
  - Shared-variance Gaussian KL: KL(ref || new) = mean ||mu_new - mu_ref||^2 / (2 * sigma^2)

References:
  - Flow-GRPO / flow-policy RL: https://arxiv.org/abs/2505.05470
"""

from __future__ import annotations


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("flow-policy optimization requires the 'train-core' extra") from error
    return torch


def shared_variance_gaussian_kl(
    new_means: object,
    reference_means: object,
    transition_scales: object,
    *,
    reduction: str = "mean",
) -> object:
    """Compute ``KL(reference || new)`` for equal diagonal covariance."""

    torch = _require_torch()
    if not all(torch.is_tensor(value) for value in (new_means, reference_means, transition_scales)):
        raise TypeError("Gaussian KL inputs must be torch.Tensor values")
    if new_means.shape != reference_means.shape or new_means.ndim < 3:
        raise ValueError("new/reference means must share shape [B,K,...]")
    scale = transition_scales.to(device=new_means.device, dtype=torch.float32)
    try:
        scale = torch.broadcast_to(scale, new_means.shape)
    except RuntimeError as error:
        raise ValueError("transition_scales cannot broadcast to means") from error
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("transition_scales must be finite and positive")
    elementwise = (new_means.float() - reference_means.float()).square() / (2.0 * scale.square())
    per_transition = elementwise.mean(dim=tuple(range(2, elementwise.ndim)))
    if reduction == "none":
        return per_transition
    if reduction == "mean":
        return per_transition.mean()
    raise ValueError("reduction must be 'none' or 'mean'")


__all__ = ["shared_variance_gaussian_kl"]
