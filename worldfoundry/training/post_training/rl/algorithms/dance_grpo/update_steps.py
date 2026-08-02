"""Per-sample random update-timestep selection used by DANCE."""

from __future__ import annotations

from math import isfinite

import torch


def sample_dance_update_step_mask(
    *,
    batch_size: int,
    transition_count: int,
    timestep_fraction: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select an independent fixed-size transition subset for every sample."""

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(transition_count, bool) or int(transition_count) <= 0:
        raise ValueError("transition_count must be positive")
    fraction = float(timestep_fraction)
    if not isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("timestep_fraction must be finite and in (0,1]")
    selected_count = int(int(transition_count) * fraction)
    if selected_count <= 0:
        raise ValueError("timestep_fraction selects no update transition")
    permutations = torch.stack(
        [
            torch.randperm(
                int(transition_count),
                device=device,
                generator=generator,
            )
            for _ in range(int(batch_size))
        ]
    )
    mask = torch.zeros(
        (int(batch_size), int(transition_count)),
        device=device,
        dtype=torch.bool,
    )
    mask.scatter_(1, permutations[:, :selected_count], True)
    return mask


__all__ = ["sample_dance_update_step_mask"]
