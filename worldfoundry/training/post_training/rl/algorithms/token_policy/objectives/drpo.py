"""Token-level smooth quadratic policy objective."""

from __future__ import annotations

import torch

from .common import (
    TokenObjective,
    validated_policy_inputs,
    validated_positive_float,
)


def token_drpo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon: float,
    mu_weighted: bool = True,
) -> TokenObjective:
    """Apply either probability-weighted or fixed-threshold SPO regularization."""

    new_logp, old_logp, advantage = validated_policy_inputs(
        new_log_probs,
        old_log_probs,
        advantages,
    )
    threshold = validated_positive_float(epsilon, field_name="epsilon")
    log_ratio = (new_logp - old_logp).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    old_probability = torch.exp(old_logp).detach()
    if mu_weighted:
        penalty_weight = old_probability
        adaptive_epsilon = torch.where(
            old_probability > 0,
            threshold / old_probability,
            torch.full_like(old_probability, float("inf")),
        )
    else:
        penalty_weight = torch.ones_like(old_probability)
        adaptive_epsilon = torch.full_like(old_probability, threshold)
    regularizer = advantage.abs() * penalty_weight * (ratio - 1.0).square() / (2.0 * threshold)
    losses = -advantage * ratio + regularizer
    return TokenObjective(
        losses=losses,
        ratio=ratio,
        log_ratio=log_ratio,
        metrics={
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
            "drpo_penalty_mean": regularizer.mean(),
            "clipfrac_upper": (ratio > 1.0 + adaptive_epsilon).float().mean(),
            "clipfrac_lower": (ratio < 1.0 - adaptive_epsilon).float().mean(),
        },
        regularizer=regularizer,
    )


__all__ = ["token_drpo_objective"]
