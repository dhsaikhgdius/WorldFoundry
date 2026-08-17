"""Clipped policy and value objectives with exact packed-token reductions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

TOKEN_MEAN = "token-mean"
SEQUENCE_MEAN_TOKEN_MEAN = "seq-mean-token-mean"
SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED = "seq-mean-token-sum-norm"
TOKEN_PPO_REDUCTIONS = frozenset(
    {
        TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
    }
)


@dataclass(frozen=True, slots=True)
class TokenPPOLossTerms:
    """Differentiable PPO numerators sharing one exact denominator."""

    policy_numerator: torch.Tensor
    value_numerator: torch.Tensor
    denominator: int
    ratio: torch.Tensor
    policy_clipped: torch.Tensor
    value_clipped: torch.Tensor

    @property
    def policy_loss(self) -> torch.Tensor:
        return self.policy_numerator / float(self.denominator)

    @property
    def value_loss(self) -> torch.Tensor:
        return self.value_numerator / float(self.denominator)


def clipped_policy_losses(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
    clip_range_high: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return elementwise ``-min(r A, clip(r) A)`` and clip indicators."""

    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    unclipped = ratio * advantages.detach()
    high = float(clip_range) if clip_range_high is None else float(clip_range_high)
    clipped_ratio = ratio.clamp(1.0 - float(clip_range), 1.0 + high)
    clipped = clipped_ratio * advantages.detach()
    return -torch.minimum(unclipped, clipped), ratio, ratio.ne(clipped_ratio)


def clipped_value_losses(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``0.5 * max((V-R)^2, (V_clip-R)^2)`` per token."""

    values_f = values.float()
    old = old_values.detach().to(device=values.device, dtype=torch.float32)
    targets = returns.detach().to(device=values.device, dtype=torch.float32)
    clipped_values = old + (values_f - old).clamp(
        -float(clip_range),
        float(clip_range),
    )
    plain = (values_f - targets).square()
    clipped = (clipped_values - targets).square()
    return (
        0.5 * torch.maximum(plain, clipped),
        values_f.detach().sub(old).abs().gt(float(clip_range)),
    )


def _reduced_numerator(
    losses: torch.Tensor,
    lengths: torch.Tensor,
    active: torch.Tensor,
    *,
    reduction: str,
    horizon: int,
) -> tuple[torch.Tensor, int]:
    selected = losses[active]
    if reduction == TOKEN_MEAN:
        return selected.sum(), int(selected.numel())

    loss_chunks = torch.split(losses, lengths.tolist())
    mask_chunks = torch.split(active, lengths.tolist())
    sequence_losses: list[torch.Tensor] = []
    for loss_chunk, mask_chunk in zip(loss_chunks, mask_chunks, strict=True):
        sequence = loss_chunk[mask_chunk]
        if reduction == SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED:
            sequence_losses.append(sequence.sum() / float(horizon))
        else:
            sequence_losses.append(sequence.mean() if sequence.numel() else loss_chunk.new_zeros(()))
    return torch.stack(sequence_losses).sum(), len(sequence_losses)


def token_ppo_loss(
    new_log_probs: torch.Tensor,
    values: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    lengths: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None,
    clip_range: float,
    clip_range_high: float | None,
    value_clip_range: float,
    reduction: str,
    horizon: int,
) -> TokenPPOLossTerms:
    """Build classic PPO actor and critic losses over a packed microbatch."""

    tensors = (values, old_log_probs, old_values, advantages, returns)
    if new_log_probs.ndim != 1 or any(tensor.shape != new_log_probs.shape for tensor in tensors):
        raise ValueError("PPO policy, value, anchor, advantage, and return tensors must align")
    if int(lengths.sum().item()) != int(new_log_probs.shape[0]):
        raise ValueError("lengths must sum to the replayed token count")
    active = (
        torch.ones_like(new_log_probs, dtype=torch.bool)
        if loss_mask is None
        else loss_mask.to(device=new_log_probs.device, dtype=torch.bool)
    )
    if active.shape != new_log_probs.shape or not bool(active.any()):
        raise ValueError("PPO loss mask must select at least one replayed token")
    policy_losses, ratio, policy_clipped = clipped_policy_losses(
        new_log_probs,
        old_log_probs,
        advantages,
        clip_range=clip_range,
        clip_range_high=clip_range_high,
    )
    value_losses, value_clipped = clipped_value_losses(
        values,
        old_values,
        returns,
        clip_range=value_clip_range,
    )
    policy_numerator, denominator = _reduced_numerator(
        policy_losses,
        lengths,
        active,
        reduction=reduction,
        horizon=horizon,
    )
    value_numerator, value_denominator = _reduced_numerator(
        value_losses,
        lengths,
        active,
        reduction=reduction,
        horizon=horizon,
    )
    if denominator != value_denominator or denominator <= 0:
        raise ValueError("PPO reduction selected no trainable units")
    return TokenPPOLossTerms(
        policy_numerator=policy_numerator,
        value_numerator=value_numerator,
        denominator=denominator,
        ratio=ratio[active],
        policy_clipped=policy_clipped[active],
        value_clipped=value_clipped[active],
    )


def token_ppo_reduction_weight(
    lengths: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    reduction: str,
) -> int:
    """Return the exact denominator for a packed PPO reduction."""

    if reduction == TOKEN_MEAN:
        return int(lengths.sum().item()) if loss_mask is None else int(loss_mask.sum().item())
    if reduction in {
        SEQUENCE_MEAN_TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
    }:
        return int(lengths.shape[0])
    raise ValueError(f"unsupported token PPO reduction: {reduction!r}")


__all__ = [
    "SEQUENCE_MEAN_TOKEN_MEAN",
    "SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED",
    "TOKEN_MEAN",
    "TOKEN_PPO_REDUCTIONS",
    "TokenPPOLossTerms",
    "clipped_policy_losses",
    "clipped_value_losses",
    "token_ppo_loss",
    "token_ppo_reduction_weight",
]
