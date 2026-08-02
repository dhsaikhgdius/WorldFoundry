"""Deterministic sliding-window stochastic-step selection."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


@dataclass(frozen=True, slots=True)
class FlowSDEWindowSchedule:
    """Move a contiguous SDE window using committed rollout ids.

    ``stride`` is the number of transition indices by which the window start
    moves. A stride of one with a four-step window therefore overlaps three
    transitions. An omitted stride uses non-overlapping windows, matching
    UniRL's ``overlap_size=0`` default. Rollback cycles through complete
    windows; otherwise the last complete window remains active.
    """

    transition_count: int
    window_size: int
    iterations_per_window: int
    stride: int | None = None
    initial_index: int = 0
    rollback: bool = False

    def __post_init__(self) -> None:
        transition_count = _positive_int(
            self.transition_count,
            field_name="transition_count",
        )
        window_size = _positive_int(self.window_size, field_name="window_size")
        iterations = _positive_int(
            self.iterations_per_window,
            field_name="iterations_per_window",
        )
        stride = window_size if self.stride is None else _positive_int(
            self.stride,
            field_name="stride",
        )
        initial = _non_negative_int(self.initial_index, field_name="initial_index")
        if not isinstance(self.rollback, bool):
            raise TypeError("rollback must be a bool")
        if initial + window_size > transition_count:
            raise ValueError("initial window exceeds the transition schedule")
        if stride > window_size:
            raise ValueError("window stride cannot exceed window_size")
        object.__setattr__(self, "transition_count", transition_count)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "iterations_per_window", iterations)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "initial_index", initial)

    @property
    def window_count(self) -> int:
        remaining = self.transition_count - self.initial_index - self.window_size
        return max(1, remaining // self.stride + 1)

    @property
    def identity(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": "worldfoundry-flow-sde-window-schedule",
                "transition_count": self.transition_count,
                "window_size": self.window_size,
                "iterations_per_window": self.iterations_per_window,
                "stride": self.stride,
                "initial_index": self.initial_index,
                "rollback": self.rollback,
            }
        )

    def resolve(self, rollout_id: int) -> tuple[int, ...]:
        rollout = _non_negative_int(rollout_id, field_name="rollout_id")
        window_step = rollout // self.iterations_per_window
        if self.rollback:
            window_step %= self.window_count
        else:
            window_step = min(window_step, self.window_count - 1)
        start = self.initial_index + window_step * self.stride
        return tuple(range(start, start + self.window_size))


__all__ = ["FlowSDEWindowSchedule"]
