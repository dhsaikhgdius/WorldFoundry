"""Pure terminal-reward and generalized-advantage calculations for token PPO."""

from __future__ import annotations

import torch


def scatter_terminal_rewards(
    rewards: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Place each sequence reward on its final packed response token."""

    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1 or not rewards.is_floating_point():
        raise TypeError("rewards must be a floating tensor with shape [B]")
    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise TypeError("lengths must be a tensor with shape [B]")
    if int(rewards.shape[0]) != int(lengths.shape[0]):
        raise ValueError("rewards and lengths must have the same batch size")
    if not bool((lengths > 0).all()):
        raise ValueError("terminal reward scattering requires positive sequence lengths")
    ends = lengths.to(device=rewards.device, dtype=torch.long).cumsum(dim=0) - 1
    token_rewards = rewards.new_zeros(int(lengths.sum().item()))
    token_rewards[ends] = rewards
    return token_rewards


def packed_gae(
    token_rewards: torch.Tensor,
    old_values: torch.Tensor,
    lengths: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE independently inside each packed sequence."""

    if token_rewards.shape != old_values.shape or token_rewards.ndim != 1:
        raise ValueError("token_rewards and old_values must be aligned one-dimensional tensors")
    if int(lengths.sum().item()) != int(token_rewards.shape[0]):
        raise ValueError("lengths must sum to the packed token count")
    if not 0.0 <= float(gamma) <= 1.0 or not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0,1]")

    rewards = token_rewards.float()
    values = old_values.detach().float()
    advantages = values.new_zeros(values.shape)
    offset = 0
    for raw_length in lengths.tolist():
        length = int(raw_length)
        end = offset + length
        running = values.new_zeros(())
        for index in range(end - 1, offset - 1, -1):
            next_value = values[index + 1] if index + 1 < end else values.new_zeros(())
            delta = rewards[index] + float(gamma) * next_value - values[index]
            running = delta + float(gamma) * float(gae_lambda) * running
            advantages[index] = running
        offset = end
    return advantages, advantages + values


__all__ = ["packed_gae", "scatter_terminal_rewards"]
