"""Tensor objective for DDRL transition replay."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from ...objectives.group_advantages import (
    normalize_data_parallel_grouped_advantages,
    normalize_grouped_advantages,
)
from ..flow_grpo.objective import clipped_policy_loss

DDRL_ADVANTAGE_EPSILON = 1.0e-4


@dataclass(frozen=True, slots=True)
class DDRLAdvantages:
    advantages: torch.Tensor
    group_means: torch.Tensor
    group_stds: torch.Tensor
    group_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DDRLLoss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor | None
    data_loss: torch.Tensor | None
    log_ratio_elements: torch.Tensor
    log_ratio: torch.Tensor
    ratio: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


def _floating_tensor(
    value: object,
    *,
    field_name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{field_name} must be a floating torch.Tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{field_name} must be finite")
    return value


def _non_negative_float(value: float, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return resolved


def ddrl_group_advantages(
    rewards: torch.Tensor,
    group_ids: tuple[str, ...],
    *,
    epsilon: float = DDRL_ADVANTAGE_EPSILON,
    normalization: str = "group-sample-std",
    clip_min: float | None = None,
    clip_max: float | None = None,
    exponential: bool = False,
    parallel_context: object | None = None,
) -> DDRLAdvantages:
    """Normalize grouped rewards and apply the configured reward transform."""

    if not isinstance(rewards, torch.Tensor) or not rewards.is_floating_point():
        raise TypeError("rewards must be a floating torch.Tensor")
    if parallel_context is None:
        grouped = normalize_grouped_advantages(
            rewards,
            group_ids,
            epsilon=epsilon,
            normalization=normalization,
        )
    else:
        grouped = normalize_data_parallel_grouped_advantages(
            rewards,
            group_ids,
            parallel_context=parallel_context,
            epsilon=epsilon,
            normalization=normalization,
        )
    advantages = grouped.advantages.detach().float()
    if not isinstance(exponential, bool):
        raise TypeError("exponential must be a bool")
    if exponential:
        advantages = -torch.exp(-advantages)
    lower = -torch.inf if clip_min is None else float(clip_min)
    upper = torch.inf if clip_max is None else float(clip_max)
    if not isfinite(lower) and lower != -torch.inf:
        raise ValueError("clip_min must be finite or None")
    if not isfinite(upper) and upper != torch.inf:
        raise ValueError("clip_max must be finite or None")
    if lower >= upper:
        raise ValueError("clip_min must be smaller than clip_max")
    advantages = advantages.clamp(min=lower, max=upper)
    return DDRLAdvantages(
        advantages=advantages,
        group_means=grouped.group_means,
        group_stds=grouped.group_stds,
        group_order=grouped.group_order,
    )


def ddrl_loss(
    *,
    next_latents: torch.Tensor,
    current_means: torch.Tensor,
    old_means: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    reference_means: torch.Tensor | None = None,
    data_loss: torch.Tensor | None = None,
    kl_beta: float = 0.0,
    data_beta: float = 0.0,
) -> DDRLLoss:
    """Compute one selected transition loss without variance normalization."""

    next_state = _floating_tensor(next_latents, field_name="next_latents")
    if next_state.ndim < 2 or int(next_state.shape[0]) == 0:
        raise ValueError("next_latents must have non-empty shape [B,...latent]")
    shape = tuple(next_state.shape)
    current = _floating_tensor(current_means, field_name="current_means", shape=shape)
    old = _floating_tensor(old_means, field_name="old_means", shape=shape)
    advantage = _floating_tensor(
        advantages,
        field_name="advantages",
        shape=(int(next_state.shape[0]),),
    )
    if any(value.device != current.device for value in (next_state, old, advantage)):
        raise ValueError("DDRL policy tensors must share a device")
    resolved_kl_beta = _non_negative_float(kl_beta, field_name="kl_beta")
    resolved_data_beta = _non_negative_float(data_beta, field_name="data_beta")

    next_for_loss = next_state.detach().float()
    old_for_loss = old.detach().to(device=current.device, dtype=torch.float32)
    current_for_loss = current.float()
    log_ratio_elements = -(next_for_loss - current_for_loss).square() + (next_for_loss - old_for_loss).square()
    reduce_dims = tuple(range(1, next_state.ndim))
    log_ratio = log_ratio_elements.mean(dim=reduce_dims)
    clipped = clipped_policy_loss(
        log_ratio.unsqueeze(1),
        torch.zeros_like(log_ratio).unsqueeze(1),
        advantage.detach().float(),
        clip_range=clip_range,
    )
    policy_loss = clipped.loss

    reference_kl: torch.Tensor | None = None
    if resolved_kl_beta > 0:
        if reference_means is None:
            raise ValueError("positive kl_beta requires reference_means")
        reference = _floating_tensor(
            reference_means,
            field_name="reference_means",
            shape=shape,
        )
        if reference.device != current.device:
            raise ValueError("reference_means must share the policy device")
        reference_kl = (current_for_loss - reference.detach().float()).square().mean()
    elif reference_means is not None:
        reference = _floating_tensor(
            reference_means,
            field_name="reference_means",
            shape=shape,
        )
        if reference.device != current.device:
            raise ValueError("reference_means must share the policy device")

    reduced_data_loss: torch.Tensor | None = None
    if resolved_data_beta > 0:
        data = _floating_tensor(data_loss, field_name="data_loss")
        if data.numel() == 0 or data.device != current.device:
            raise ValueError("data_loss must be non-empty and share the policy device")
        reduced_data_loss = data.float().mean()
    elif data_loss is not None:
        data = _floating_tensor(data_loss, field_name="data_loss")
        if data.numel() == 0 or data.device != current.device:
            raise ValueError("data_loss must be non-empty and share the policy device")

    loss = policy_loss
    if reference_kl is not None:
        loss = loss + resolved_kl_beta * reference_kl
    if reduced_data_loss is not None:
        loss = loss + resolved_data_beta * reduced_data_loss
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("non-finite DDRL objective")
    return DDRLLoss(
        loss=loss,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        data_loss=reduced_data_loss,
        log_ratio_elements=log_ratio_elements,
        log_ratio=log_ratio,
        ratio=clipped.ratio.squeeze(1),
        clip_fraction=clipped.clip_fraction,
        approx_kl=clipped.approx_kl,
    )


__all__ = [
    "DDRL_ADVANTAGE_EPSILON",
    "DDRLAdvantages",
    "DDRLLoss",
    "ddrl_group_advantages",
    "ddrl_loss",
]
