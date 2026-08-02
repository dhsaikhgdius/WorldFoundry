"""Token-level GRPO clipped policy objective."""

from __future__ import annotations

import torch

from .common import TokenObjective, clipped_policy_objective


def token_grpo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
    clip_range_high: float | None = None,
) -> TokenObjective:
    """Apply symmetric or clip-higher PPO clipping per packed token."""

    return clipped_policy_objective(
        new_log_probs,
        old_log_probs,
        advantages,
        clip_range=clip_range,
        clip_range_high=clip_range_high,
    )


__all__ = ["token_grpo_objective"]
