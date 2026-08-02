"""Advantage normalization for prompt-grouped native RL rollouts.

Key formulas:
  - Group mean: mu_g = mean(r_i | group_i = g)
  - Group advantage (population std): A_i = (r_i - mu_g) / (std_g + eps)
  - Group advantage (population variance): A_i = (r_i - mu_g) / (var_g + eps)
  - Weighted components: A = sum_k w_k * A_k with normalized weights

References:
  - GRPO (group-relative baseline): https://arxiv.org/abs/2402.03300
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from worldfoundry.training.recipes.post_training.common import (
    advantage_normalization_mode,
)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("post-training advantage math requires the 'train-core' extra") from error
    return torch


@dataclass(frozen=True, slots=True)
class GroupAdvantageResult:
    advantages: object
    group_means: object
    group_stds: object
    group_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WeightedComponentAdvantageResult:
    """Component-first grouped advantages and their normalized weights."""

    advantages: object
    components: Mapping[str, GroupAdvantageResult]
    normalized_weights: Mapping[str, float]


def normalize_grouped_advantages(
    rewards: object,
    group_ids: tuple[str, ...],
    *,
    epsilon: float = 1.0e-8,
    clip_max: float | None = None,
    normalization: str = "group-population-variance",
    global_standard_deviation: object | None = None,
) -> GroupAdvantageResult:
    """Normalize scalar rewards with an explicit source-level denominator."""

    torch = _require_torch()
    if not torch.is_tensor(rewards) or rewards.ndim != 1:
        raise TypeError("rewards must be a one-dimensional torch.Tensor")
    if len(group_ids) != int(rewards.shape[0]):
        raise ValueError("group_ids length must match rewards")
    if not group_ids or any(not isinstance(group, str) or not group.strip() for group in group_ids):
        raise ValueError("group_ids must contain non-empty strings")
    resolved_epsilon = float(epsilon)
    if not isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if clip_max is not None and (not isfinite(float(clip_max)) or float(clip_max) <= 0):
        raise ValueError("clip_max must be finite and positive")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")
    mode = advantage_normalization_mode(
        normalization,
        field_name="normalization",
    )

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        grouped[group].append(index)
    incomplete = sorted(group for group, indices in grouped.items() if len(indices) < 2)
    if incomplete:
        raise ValueError(f"every advantage group must contain at least two rewards: {incomplete}")

    rewards_fp32 = rewards.float()
    supplied_global_std = None
    if global_standard_deviation is not None:
        if mode not in {
            "group-mean-global-sample-std",
            "group-mean-global-population-std",
        }:
            raise ValueError("global_standard_deviation is valid only for a group-mean-global normalization")
        if not torch.is_tensor(global_standard_deviation) or global_standard_deviation.numel() != 1:
            raise TypeError("global_standard_deviation must be a scalar torch.Tensor")
        supplied_global_std = global_standard_deviation.detach().to(
            device=rewards.device,
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(supplied_global_std)) or bool(supplied_global_std < 0):
            raise ValueError("global_standard_deviation must be finite and non-negative")
    if mode == "group-mean-global-sample-std":
        global_reported_std = rewards_fp32.std(correction=1) if supplied_global_std is None else supplied_global_std
        global_denominator = global_reported_std + resolved_epsilon
    elif mode == "group-mean-global-population-std":
        global_reported_std = rewards_fp32.std(correction=0) if supplied_global_std is None else supplied_global_std
        global_denominator = global_reported_std + resolved_epsilon
    else:
        global_denominator = None
        global_reported_std = None
    advantages = torch.zeros_like(rewards_fp32)
    means: list[object] = []
    stds: list[object] = []
    order = tuple(grouped)
    for group in order:
        indices = torch.tensor(grouped[group], device=rewards.device, dtype=torch.long)
        values = rewards_fp32.index_select(0, indices)
        mean = values.mean()
        if global_denominator is not None:
            denominator = global_denominator
            assert global_reported_std is not None
            reported_std = global_reported_std
        elif mode == "group-population-variance":
            variance = values.var(correction=0)
            denominator = (variance + resolved_epsilon).sqrt()
            reported_std = variance.sqrt()
        elif mode == "group-population-std":
            reported_std = values.std(correction=0)
            denominator = reported_std + resolved_epsilon
        else:
            reported_std = values.std(correction=1)
            denominator = reported_std + resolved_epsilon
        if not bool(torch.isfinite(denominator)) or bool(denominator <= 0):
            raise FloatingPointError("advantage normalization denominator is not finite and positive")
        normalized = (values - mean) / denominator
        advantages.index_copy_(0, indices, normalized)
        means.append(mean)
        stds.append(reported_std)

    if clip_max is not None:
        advantages = advantages.clamp(min=-float(clip_max), max=float(clip_max))
    return GroupAdvantageResult(
        advantages=advantages.to(dtype=rewards.dtype),
        group_means=torch.stack(means),
        group_stds=torch.stack(stds),
        group_order=order,
    )


def normalize_data_parallel_grouped_advantages(
    rewards: object,
    group_ids: tuple[str, ...],
    *,
    parallel_context: object,
    epsilon: float = 1.0e-8,
    clip_max: float | None = None,
    normalization: str = "group-population-variance",
) -> GroupAdvantageResult:
    """Normalize local prompt groups with the declared data-parallel statistics."""

    mode = advantage_normalization_mode(normalization, field_name="normalization")
    correction = {
        "group-mean-global-population-std": 0,
        "group-mean-global-sample-std": 1,
    }.get(mode)
    global_std = None
    if correction is not None:
        compute = getattr(parallel_context, "global_standard_deviation", None)
        if not callable(compute):
            raise TypeError("parallel_context must compute global_standard_deviation for global normalization")
        global_std = compute(rewards, correction=correction)
    return normalize_grouped_advantages(
        rewards,
        group_ids,
        epsilon=epsilon,
        clip_max=clip_max,
        normalization=mode,
        global_standard_deviation=global_std,
    )


def normalize_weighted_component_advantages(
    rewards: Mapping[str, object],
    weights: Mapping[str, float],
    group_ids: tuple[str, ...],
    *,
    parallel_context: object,
    epsilon: float = 1.0e-8,
    clip_max: float | None = None,
    normalization: str = "group-sample-std",
) -> WeightedComponentAdvantageResult:
    """Normalize each reward component by prompt group, then mix advantages.

    Weight normalization happens before aggregation.  Clipping applies only to
    the merged advantage, matching the component-first MixGRPO objective.
    """

    torch = _require_torch()
    if not isinstance(rewards, Mapping) or not rewards:
        raise ValueError("reward components must be a non-empty mapping")
    resolved_rewards = {str(name): value for name, value in rewards.items()}
    resolved_weights = {str(name): float(value) for name, value in weights.items()}
    if set(resolved_rewards) != set(resolved_weights):
        raise ValueError("component reward and weight keys must match exactly")
    if any(not isfinite(value) or value < 0 for value in resolved_weights.values()):
        raise ValueError("component advantage weights must be finite and non-negative")
    weight_sum = sum(resolved_weights.values())
    if not isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("at least one component advantage weight must be positive")
    normalized_weights = {
        name: value / weight_sum for name, value in resolved_weights.items()
    }

    component_results: dict[str, GroupAdvantageResult] = {}
    merged = None
    for name, values in resolved_rewards.items():
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != (len(group_ids),):
            raise ValueError(f"reward component {name!r} must have shape [B]")
        result = normalize_data_parallel_grouped_advantages(
            values,
            group_ids,
            parallel_context=parallel_context,
            epsilon=epsilon,
            clip_max=None,
            normalization=normalization,
        )
        component_results[name] = result
        weighted = result.advantages.float() * normalized_weights[name]
        merged = weighted if merged is None else merged + weighted
    assert merged is not None
    if clip_max is not None:
        resolved_clip = float(clip_max)
        if not isfinite(resolved_clip) or resolved_clip <= 0:
            raise ValueError("clip_max must be finite and positive")
        merged = merged.clamp(min=-resolved_clip, max=resolved_clip)
    return WeightedComponentAdvantageResult(
        advantages=merged,
        components=MappingProxyType(component_results),
        normalized_weights=MappingProxyType(normalized_weights),
    )


__all__ = [
    "GroupAdvantageResult",
    "WeightedComponentAdvantageResult",
    "normalize_data_parallel_grouped_advantages",
    "normalize_grouped_advantages",
    "normalize_weighted_component_advantages",
]
