"""Sparse stochastic-step selection for flow-policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FlowSDEIndexSchedule:
    """Resolve the stochastic transition indices for one rollout.

    The schedule is either an explicit set of indices or a deterministic draw
    without replacement from a fractional transition window.  Dynamic draws
    are a pure function of ``rollout_id``, so there is no mutable state to
    checkpoint.
    """

    transition_count: int
    static_indices: tuple[int, ...] | None = None
    timestep_fraction: tuple[float, float] | None = None
    num_sde_steps: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.transition_count, bool) or int(self.transition_count) <= 0:
            raise ValueError("transition_count must be a positive integer")
        count = int(self.transition_count)
        static = self.static_indices
        fraction = self.timestep_fraction
        sparse_count = self.num_sde_steps
        if static is not None:
            indices = tuple(int(value) for value in static)
            if not indices or indices != tuple(sorted(set(indices))) or indices[0] < 0 or indices[-1] >= count:
                raise ValueError("static SDE indices must be non-empty, sorted, unique, and in range")
            if fraction is not None or sparse_count is not None:
                raise ValueError("static SDE indices cannot be combined with a fractional schedule")
            object.__setattr__(self, "transition_count", count)
            object.__setattr__(self, "static_indices", indices)
            return
        if fraction is None or sparse_count is None:
            raise ValueError("dynamic SDE selection requires timestep_fraction and num_sde_steps")
        values = tuple(float(value) for value in fraction)
        if (
            len(values) != 2
            or any(not isfinite(value) or not 0 <= value <= 1 for value in values)
            or values[0] > values[1]
        ):
            raise ValueError("timestep_fraction must be an ordered pair in [0,1]")
        if isinstance(sparse_count, bool) or int(sparse_count) <= 0:
            raise ValueError("num_sde_steps must be a positive integer")
        start = int(count * values[0])
        end = int(count * values[1])
        pool_size = end - start
        if pool_size <= 0 or int(sparse_count) > pool_size:
            raise ValueError("num_sde_steps exceeds the non-empty fractional timestep window")
        object.__setattr__(self, "transition_count", count)
        object.__setattr__(self, "timestep_fraction", values)
        object.__setattr__(self, "num_sde_steps", int(sparse_count))

    @property
    def identity(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": "worldfoundry-flow-sde-index-schedule",
                "transition_count": self.transition_count,
                "static_indices": self.static_indices,
                "timestep_fraction": self.timestep_fraction,
                "num_sde_steps": self.num_sde_steps,
            }
        )

    def resolve(self, rollout_id: int) -> tuple[int, ...]:
        if isinstance(rollout_id, bool) or int(rollout_id) < 0:
            raise ValueError("rollout_id must be a non-negative integer")
        if self.static_indices is not None:
            return self.static_indices
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise RuntimeError("dynamic SDE index selection requires the 'train-core' extra") from error
        assert self.timestep_fraction is not None and self.num_sde_steps is not None
        start = int(self.transition_count * self.timestep_fraction[0])
        end = int(self.transition_count * self.timestep_fraction[1])
        rng = np.random.default_rng(int(rollout_id))
        selected = rng.choice(
            np.arange(start, end),
            size=self.num_sde_steps,
            replace=False,
        )
        return tuple(sorted(int(value) for value in selected.tolist()))


__all__ = ["FlowSDEIndexSchedule"]
