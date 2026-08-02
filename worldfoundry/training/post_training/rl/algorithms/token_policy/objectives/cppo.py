"""Position-weighted cumulative-prefix Binary-TV objective."""

from __future__ import annotations

from math import isfinite

import torch

from .common import (
    TokenObjective,
    validated_policy_inputs,
    validated_positive_float,
)


def token_cppo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    lengths: torch.Tensor,
    *,
    delta: float,
    w_min: float,
    delta_b: float,
) -> TokenObjective:
    """Gate diverging token updates by position and accumulated prefix drift."""

    new_logp, old_logp, advantage = validated_policy_inputs(
        new_log_probs,
        old_log_probs,
        advantages,
    )
    threshold = validated_positive_float(delta, field_name="delta")
    position_floor = float(w_min)
    prefix_floor = float(delta_b)
    if not isfinite(position_floor) or not 0 < position_floor <= 1:
        raise ValueError("w_min must be finite and in (0,1]")
    if not isfinite(prefix_floor) or prefix_floor < 0:
        raise ValueError("delta_b must be finite and non-negative")
    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise TypeError("lengths must be a one-dimensional tensor")
    if not bool((lengths >= 0).all()):
        raise ValueError("lengths must be non-negative")
    if int(lengths.sum().item()) != int(new_logp.shape[0]):
        raise ValueError("lengths must sum to the packed token count")

    log_ratio = (new_logp - old_logp).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    with torch.no_grad():
        divergence = (torch.exp(new_logp.float()) - torch.exp(old_logp.float())).abs()
        toward_old = advantage * (ratio - 1.0) <= 0
        keep_parts: list[torch.Tensor] = []
        divergence_parts = torch.split(divergence, lengths.tolist())
        toward_parts = torch.split(toward_old, lengths.tolist())
        for sequence_divergence, sequence_toward in zip(
            divergence_parts,
            toward_parts,
        ):
            token_count = int(sequence_divergence.shape[0])
            if token_count == 0:
                keep_parts.append(sequence_divergence.new_zeros(0, dtype=torch.bool))
                continue
            positions = torch.arange(
                token_count,
                device=sequence_divergence.device,
                dtype=sequence_divergence.dtype,
            )
            position_weight = 1.0 - (1.0 - position_floor) * positions / max(
                token_count - 1,
                1,
            )
            weighted_divergence = position_weight * sequence_divergence
            previous_divergence = torch.cat(
                [
                    weighted_divergence.new_zeros(1),
                    weighted_divergence.cumsum(dim=0)[:-1],
                ]
            )
            previous_weight = torch.cat(
                [
                    position_weight.new_zeros(1),
                    position_weight.cumsum(dim=0)[:-1],
                ]
            )
            sequence_prefix_budget = torch.quantile(
                sequence_divergence,
                q=0.9,
            ).clamp(min=prefix_floor, max=2.0 * prefix_floor)
            effective_threshold = torch.minimum(
                torch.full_like(weighted_divergence, threshold),
                threshold + sequence_prefix_budget * previous_weight - previous_divergence,
            )
            keep_parts.append(sequence_toward | (weighted_divergence <= effective_threshold))
        keep = torch.cat(keep_parts).to(dtype=new_logp.dtype)
    losses = -advantage * ratio * keep
    return TokenObjective(
        losses=losses,
        ratio=ratio,
        log_ratio=log_ratio,
        metrics={
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
            "masked_fraction": (1.0 - keep).mean(),
        },
        keep_mask=keep,
    )


__all__ = ["token_cppo_objective"]
