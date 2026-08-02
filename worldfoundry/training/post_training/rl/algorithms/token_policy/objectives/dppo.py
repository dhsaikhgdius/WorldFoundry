"""Token-level Binary-TV hard-mask objective."""

from __future__ import annotations

import torch

from .common import (
    TokenObjective,
    validated_policy_inputs,
    validated_positive_float,
)


def token_dppo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    delta: float,
) -> TokenObjective:
    """Keep corrective updates or probability shifts within a uniform budget."""

    new_logp, old_logp, advantage = validated_policy_inputs(
        new_log_probs,
        old_log_probs,
        advantages,
    )
    threshold = validated_positive_float(delta, field_name="delta")
    log_ratio = (new_logp - old_logp).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    with torch.no_grad():
        divergence = (torch.exp(new_logp.float()) - torch.exp(old_logp.float())).abs()
        toward_old = advantage * (ratio - 1.0) <= 0
        keep = (toward_old | (divergence <= threshold)).to(dtype=new_logp.dtype)
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


__all__ = ["token_dppo_objective"]
