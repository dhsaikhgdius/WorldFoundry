"""Packed token indexing shared by replay, objectives, and microbatching."""

from __future__ import annotations

from collections.abc import Mapping

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


def select_packed_token_trajectory(
    trajectory: PackedTokenTrajectory,
    indices: tuple[int, ...],
) -> PackedTokenTrajectory:
    """Repack an ordered sequence subset and retain excluded sample ids."""

    selected = tuple(int(index) for index in indices)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= trajectory.batch_size for index in selected)
        or tuple(sorted(selected)) != selected
    ):
        raise ValueError("packed-token selection must be non-empty, unique, ordered, and in range")
    offsets = packed_token_offsets(trajectory.lengths)
    token_chunks = tuple(
        trajectory.tokens[int(offsets[index].item()) : int(offsets[index + 1].item())]
        for index in selected
    )
    log_prob_chunks = tuple(
        trajectory.old_log_probs[
            int(offsets[index].item()) : int(offsets[index + 1].item())
        ]
        for index in selected
    )

    def select_value(value: object) -> object:
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == trajectory.batch_size:
                positions = torch.tensor(selected, device=value.device, dtype=torch.long)
                return value.index_select(0, positions)
            return value
        if isinstance(value, tuple) and len(value) == trajectory.batch_size:
            return tuple(value[index] for index in selected)
        if isinstance(value, list) and len(value) == trajectory.batch_size:
            return [value[index] for index in selected]
        if isinstance(value, Mapping):
            return {str(key): select_value(item) for key, item in value.items()}
        return value

    selected_ids = {trajectory.sample_ids[index] for index in selected}
    newly_excluded = tuple(
        sample_id
        for sample_id in trajectory.sample_ids
        if sample_id not in selected_ids
    )
    positions = torch.tensor(selected, device=trajectory.lengths.device, dtype=torch.long)
    return PackedTokenTrajectory(
        sample_ids=tuple(trajectory.sample_ids[index] for index in selected),
        group_ids=tuple(trajectory.group_ids[index] for index in selected),
        policy_revision=trajectory.policy_revision,
        tokens=torch.cat(token_chunks),
        lengths=trajectory.lengths.index_select(0, positions),
        old_log_probs=torch.cat(log_prob_chunks),
        sampling_temperature=trajectory.sampling_temperature,
        conditioning={
            str(key): select_value(value)
            for key, value in trajectory.conditioning.items()
        },
        excluded_sample_ids=trajectory.excluded_sample_ids + newly_excluded,
    )


__all__ = [
    "expand_sequence_values",
    "packed_token_offsets",
    "select_packed_token_trajectory",
    "slice_packed_token_trajectory",
]
