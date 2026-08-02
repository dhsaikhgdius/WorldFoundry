"""Sequence-level GSPO objective over packed token log-probabilities."""

from __future__ import annotations

import torch

from .common import TokenObjective, clipped_policy_objective

MAX_SEQUENCE_LOG_RATIO = 10.0


def _sequence_means(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    sequence_count = int(lengths.shape[0])
    segment_ids = torch.repeat_interleave(
        torch.arange(sequence_count, device=values.device),
        lengths.to(device=values.device, dtype=torch.long),
    )
    denominator = lengths.to(device=values.device, dtype=values.dtype).clamp(min=1)
    return values.new_zeros(sequence_count).index_add(0, segment_ids, values) / denominator


def token_gspo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    lengths: torch.Tensor,
    *,
    clip_range: float,
    clip_range_high: float | None = None,
) -> TokenObjective:
    """Clip one ratio per non-empty sequence using its mean token log-ratio."""

    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise TypeError("lengths must be a one-dimensional tensor")
    if int(advantages.shape[0]) != int(lengths.shape[0]):
        raise ValueError("advantages and lengths must have the same sequence count")
    if not bool((lengths >= 0).all()):
        raise ValueError("lengths must be non-negative")
    if int(lengths.sum().item()) != int(new_log_probs.shape[0]):
        raise ValueError("lengths must sum to the packed token count")
    if old_log_probs.shape != new_log_probs.shape:
        raise ValueError("new_log_probs and old_log_probs must share shape [tokens]")

    valid = lengths.to(device=new_log_probs.device) > 0
    if not bool(valid.any()):
        empty = new_log_probs[:0]
        return TokenObjective(
            losses=empty,
            ratio=empty,
            log_ratio=empty,
            metrics={},
        )
    seq_new = _sequence_means(new_log_probs, lengths)[valid]
    seq_old = _sequence_means(
        old_log_probs.detach().to(
            device=new_log_probs.device,
            dtype=new_log_probs.dtype,
        ),
        lengths,
    )[valid]
    seq_advantage = advantages.detach().to(
        device=new_log_probs.device,
        dtype=new_log_probs.dtype,
    )[valid]
    log_ratio = (seq_new - seq_old).clamp(max=MAX_SEQUENCE_LOG_RATIO)
    return clipped_policy_objective(
        log_ratio,
        torch.zeros_like(log_ratio),
        seq_advantage,
        clip_range=clip_range,
        clip_range_high=clip_range_high,
    )


__all__ = ["MAX_SEQUENCE_LOG_RATIO", "token_gspo_objective"]
