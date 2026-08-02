"""Immutable request/result contracts for native reward evaluators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from ..shared.contracts import freeze_mapping


@dataclass(frozen=True, slots=True)
class RewardRequest:
    """One ordered reward request over already-produced artifacts."""

    request_id: str
    rollout_id: str
    prompt: str
    conditions: Mapping[str, object]
    artifacts: Mapping[str, object]
    reward_ids: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("rollout_id", self.rollout_id),
            ("prompt", self.prompt),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        reward_ids = tuple(str(value) for value in self.reward_ids)
        if not reward_ids or any(not value.strip() for value in reward_ids):
            raise ValueError("reward_ids must contain non-empty strings")
        if len(set(reward_ids)) != len(reward_ids):
            raise ValueError("reward_ids must be unique")
        object.__setattr__(self, "reward_ids", reward_ids)
        object.__setattr__(
            self,
            "conditions",
            freeze_mapping(self.conditions, field_name="conditions"),
        )
        object.__setattr__(
            self,
            "artifacts",
            freeze_mapping(self.artifacts, field_name="artifacts"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, field_name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class RewardResult:
    """Vector reward result with per-component validity."""

    request_id: str
    rollout_id: str
    values: Mapping[str, float]
    valid: Mapping[str, bool]
    diagnostics: Mapping[str, object]
    latency_ms: float

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("rollout_id", self.rollout_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        values = {str(key): float(value) for key, value in self.values.items()}
        valid = {str(key): value for key, value in self.valid.items()}
        if not values or set(values) != set(valid):
            raise ValueError("reward values and validity maps must have the same non-empty keys")
        if any(not isinstance(value, bool) for value in valid.values()):
            raise TypeError("reward validity values must be bool")
        if any(valid[key] and not isfinite(values[key]) for key in values):
            raise ValueError("valid rewards must be finite")
        latency = float(self.latency_ms)
        if not isfinite(latency) or latency < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "valid", MappingProxyType(valid))
        object.__setattr__(
            self,
            "diagnostics",
            freeze_mapping(self.diagnostics, field_name="diagnostics"),
        )
        object.__setattr__(self, "latency_ms", latency)


@runtime_checkable
class RewardEvaluator(Protocol):
    """Synchronous first-party seam; remote clients implement the same contract."""

    def evaluate(
        self,
        requests: tuple[RewardRequest, ...],
    ) -> tuple[RewardResult, ...]: ...


__all__ = ["RewardEvaluator", "RewardRequest", "RewardResult"]
