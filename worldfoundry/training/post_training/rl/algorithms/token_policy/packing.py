"""Packed token indexing shared by replay, objectives, and microbatching."""

from __future__ import annotations

import torch

from .contracts import PackedTokenReplayBatch, PackedTokenTrajectory


def packed_token_offsets(lengths: torch.Tensor) -> torch.Tensor:
    """Return cumulative token offsets ``[B+1]`` for non-negative lengths."""

    if (
        not isinstance(lengths, torch.Tensor)
        or lengths.ndim != 1
        or lengths.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    ):
        raise TypeError("lengths must be a one-dimensional integer tensor")
    if not bool((lengths >= 0).all()):
        raise ValueError("lengths must be non-negative")
    return torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)])


def expand_sequence_values(
    values: torch.Tensor,
    lengths: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Repeat one value per sequence over its packed token span."""

    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise TypeError("sequence values must have shape [B]")
    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise TypeError("lengths must have shape [B]")
    if int(values.shape[0]) != int(lengths.shape[0]):
        raise ValueError("sequence values and lengths must have the same batch size")
    return torch.repeat_interleave(
        values.detach().to(device=device, dtype=dtype),
        lengths.to(device=device, dtype=torch.long),
    )


def slice_packed_token_trajectory(
    trajectory: PackedTokenTrajectory,
    start: int,
    end: int,
) -> PackedTokenReplayBatch:
    """Select a contiguous sequence microbatch without padding or repacking."""

    if not isinstance(trajectory, PackedTokenTrajectory):
        raise TypeError("trajectory must be a PackedTokenTrajectory")
    if isinstance(start, bool) or isinstance(end, bool) or not 0 <= int(start) < int(end) <= trajectory.batch_size:
        raise ValueError("sequence slice must be a non-empty in-range interval")
    begin, stop = int(start), int(end)
    offsets = packed_token_offsets(trajectory.lengths)
    return PackedTokenReplayBatch(
        source=trajectory,
        sequence_start=begin,
        sequence_end=stop,
        token_start=int(offsets[begin].item()),
        token_end=int(offsets[stop].item()),
    )


__all__ = [
    "expand_sequence_values",
    "packed_token_offsets",
    "slice_packed_token_trajectory",
]
