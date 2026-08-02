"""Contracts shared by deterministic stochastic-step schedules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class FlowSDEIndexResolver(Protocol):
    transition_count: int

    @property
    def identity(self) -> Mapping[str, object]: ...

    def resolve(self, rollout_id: int) -> tuple[int, ...]: ...


__all__ = ["FlowSDEIndexResolver"]
