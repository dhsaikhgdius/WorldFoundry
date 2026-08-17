"""Trajectory-level reward aggregation for agentic policy learning."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

from ..rl.algorithms.token_policy.contracts import PackedTokenTrajectory
from .contracts import AgenticSampleTrajectory, agentic_trajectory_from_packed

RewardReduction = Literal["sum", "mean", "last"]


@dataclass(frozen=True, slots=True)
class AgenticRewardComponent:
    """Reduce scalar signals emitted for a complete sample trajectory."""

    reward_id: str
    evaluator: Callable[[AgenticSampleTrajectory], float | Iterable[float]]
    reduction: RewardReduction = "sum"

    def __post_init__(self) -> None:
        if not isinstance(self.reward_id, str) or not self.reward_id.strip():
            raise ValueError("reward_id must be a non-empty string")
        if not callable(self.evaluator):
            raise TypeError("agentic reward evaluator must be callable")
        if self.reduction not in {"sum", "mean", "last"}:
            raise ValueError("agentic reward reduction must be sum, mean, or last")

    def score(self, trajectory: AgenticSampleTrajectory) -> float:
        raw = self.evaluator(trajectory)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            values = (float(raw),)
        else:
            values = tuple(float(value) for value in raw)
        if not values or not all(isfinite(value) for value in values):
            raise ValueError(f"agentic reward {self.reward_id!r} produced invalid signals")
        if self.reduction == "sum":
            return sum(values)
        if self.reduction == "mean":
            return sum(values) / len(values)
        return values[-1]


class AgenticTrajectoryRewardAdapter:
    """Expose agentic trajectory components to the shared reward scalarizer."""

    def __init__(self, components: tuple[AgenticRewardComponent, ...]) -> None:
        resolved = tuple(components)
        reward_ids = tuple(component.reward_id for component in resolved)
        if not resolved or len(set(reward_ids)) != len(reward_ids):
            raise ValueError("agentic reward components must be non-empty with unique ids")
        self.components = resolved
        self.reward_ids = reward_ids

    def score(self, trajectory: PackedTokenTrajectory) -> Mapping[str, torch.Tensor]:
        agentic = agentic_trajectory_from_packed(trajectory)
        device = trajectory.old_log_probs.device
        return {
            component.reward_id: torch.tensor(
                [component.score(sample) for sample in agentic.samples],
                device=device,
                dtype=torch.float32,
            )
            for component in self.components
        }


__all__ = [
    "AgenticRewardComponent",
    "AgenticTrajectoryRewardAdapter",
    "RewardReduction",
]
