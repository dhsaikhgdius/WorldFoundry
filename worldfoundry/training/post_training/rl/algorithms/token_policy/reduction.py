"""Exact reductions for packed token losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

TOKEN_MEAN = "token-mean"
SEQUENCE_MEAN_TOKEN_MEAN = "seq-mean-token-mean"
SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED = "seq-mean-token-sum-norm"

TOKEN_REDUCTIONS = frozenset(
    {
        TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
    }
)
SUM_REDUCTIONS = frozenset({TOKEN_MEAN, SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED})


@dataclass(frozen=True, slots=True)
class ReducedTokenLoss:
    """A differentiable numerator and its exact averaging weight."""

    numerator: torch.Tensor
    denominator: int


def validate_reduction(
    mode: str,
    *,
    allowed: frozenset[str],
    horizon: int,
) -> tuple[str, int]:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in allowed:
        raise ValueError(f"reduction must be one of {sorted(allowed)}; got {mode!r}")
    if isinstance(horizon, bool) or int(horizon) <= 0:
        raise ValueError("horizon must be a positive integer")
    return resolved_mode, int(horizon)


def reduction_weight(lengths: torch.Tensor, *, mode: str) -> int:
    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise TypeError("lengths must have shape [B]")
    if not bool((lengths >= 0).all()):
        raise ValueError("lengths must be non-negative")
    if mode == TOKEN_MEAN:
        return int(lengths.sum().item())
    if mode in {
        SEQUENCE_MEAN_TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
    }:
        return int(lengths.shape[0])
    raise ValueError(f"unsupported token reduction: {mode!r}")


def reduce_token_losses(
    losses: torch.Tensor,
    lengths: torch.Tensor,
    *,
    mode: str,
    horizon: int,
) -> ReducedTokenLoss:
    """Reduce without losing the numerator needed for exact microbatch scaling."""

    if not isinstance(losses, torch.Tensor) or losses.ndim != 1:
        raise TypeError("losses must have shape [tokens]")
    if int(lengths.sum().item()) != int(losses.shape[0]):
        raise ValueError("lengths must sum to the packed loss count")
    denominator = reduction_weight(lengths, mode=mode)
    if mode == TOKEN_MEAN:
        return ReducedTokenLoss(losses.sum(), denominator)
    if mode == SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED:
        return ReducedTokenLoss(losses.sum() / float(horizon), denominator)
    if mode == SEQUENCE_MEAN_TOKEN_MEAN:
        parts = torch.split(losses, lengths.tolist())
        sequence_means = [part.mean() if part.numel() else losses.new_zeros(()) for part in parts]
        numerator = torch.stack(sequence_means).sum() if sequence_means else losses.sum()
        return ReducedTokenLoss(numerator, denominator)
    raise ValueError(f"unsupported token reduction: {mode!r}")


__all__ = [
    "SEQUENCE_MEAN_TOKEN_MEAN",
    "SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED",
    "SUM_REDUCTIONS",
    "TOKEN_MEAN",
    "TOKEN_REDUCTIONS",
    "ReducedTokenLoss",
    "reduce_token_losses",
    "reduction_weight",
    "validate_reduction",
]
