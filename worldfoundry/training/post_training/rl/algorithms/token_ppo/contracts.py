"""Packed autoregressive actor-critic contracts for classic PPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, runtime_checkable

import torch

from ....shared.contracts import freeze_mapping, non_empty_ids

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True, slots=True)
class TokenPPORolloutRequest:
    """One batch requested from an autoregressive behavior policy."""

    sample_ids: tuple[str, ...]
    policy_revision: str
    sampling_temperature: float = 1.0
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "sampling_temperature", temperature)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class PackedTokenPPOTrajectory:
    """Packed response tokens with behavior-policy log-probability anchors."""

    sample_ids: tuple[str, ...]
    policy_revision: str
    tokens: torch.Tensor
    lengths: torch.Tensor
    old_log_probs: torch.Tensor
    loss_mask: torch.Tensor | None = None
    sampling_temperature: float = 1.0
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        if (
            not isinstance(self.lengths, torch.Tensor)
            or self.lengths.ndim != 1
            or self.lengths.dtype not in _INTEGER_DTYPES
            or int(self.lengths.shape[0]) != len(samples)
        ):
            raise TypeError("lengths must be an integer tensor with shape [B]")
        if not bool((self.lengths > 0).all()):
            raise ValueError("PPO trajectories require at least one response token per sample")
        token_count = int(self.lengths.sum().item())
        if (
            not isinstance(self.tokens, torch.Tensor)
            or self.tokens.ndim != 1
            or self.tokens.dtype not in _INTEGER_DTYPES
            or int(self.tokens.shape[0]) != token_count
        ):
            raise TypeError("tokens must be an integer tensor with shape [sum(lengths)]")
        if (
            not isinstance(self.old_log_probs, torch.Tensor)
            or self.old_log_probs.ndim != 1
            or not self.old_log_probs.is_floating_point()
            or int(self.old_log_probs.shape[0]) != token_count
        ):
            raise TypeError("old_log_probs must be a floating tensor with shape [sum(lengths)]")
        if not bool(torch.isfinite(self.old_log_probs).all()):
            raise ValueError("old_log_probs must be finite")
        if self.loss_mask is not None:
            if (
                not isinstance(self.loss_mask, torch.Tensor)
                or self.loss_mask.dtype is not torch.bool
                or tuple(self.loss_mask.shape) != (token_count,)
            ):
                raise TypeError("loss_mask must be a boolean tensor with shape [sum(lengths)]")
            if not bool(self.loss_mask.any()):
                raise ValueError("loss_mask must select at least one token")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "sampling_temperature", temperature)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[0])


def _slice_conditioning(
    values: Mapping[str, object],
    *,
    start: int,
    end: int,
    batch_size: int,
) -> Mapping[str, object]:
    def slice_value(value: object) -> object:
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            return value[start:end]
        if isinstance(value, tuple) and len(value) == batch_size:
            return value[start:end]
        if isinstance(value, list) and len(value) == batch_size:
            return value[start:end]
        if isinstance(value, Mapping):
            return {str(key): slice_value(item) for key, item in value.items()}
        return value

    return freeze_mapping(
        {str(key): slice_value(value) for key, value in values.items()},
        field_name="conditioning",
    )


@dataclass(frozen=True, slots=True)
class PackedTokenPPOReplayBatch:
    """A contiguous sequence slice preserving the original packed token span."""

    source: PackedTokenPPOTrajectory
    sequence_start: int
    sequence_end: int
    token_start: int
    token_end: int

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.source.sample_ids[self.sequence_start : self.sequence_end]

    @property
    def tokens(self) -> torch.Tensor:
        return self.source.tokens[self.token_start : self.token_end]

    @property
    def lengths(self) -> torch.Tensor:
        return self.source.lengths[self.sequence_start : self.sequence_end]

    @property
    def old_log_probs(self) -> torch.Tensor:
        return self.source.old_log_probs[self.token_start : self.token_end]

    @property
    def loss_mask(self) -> torch.Tensor | None:
        if self.source.loss_mask is None:
            return None
        return self.source.loss_mask[self.token_start : self.token_end]

    @property
    def sampling_temperature(self) -> float:
        return self.source.sampling_temperature

    @property
    def conditioning(self) -> Mapping[str, object]:
        return _slice_conditioning(
            self.source.conditioning,
            start=self.sequence_start,
            end=self.sequence_end,
            batch_size=self.source.batch_size,
        )

    @property
    def batch_size(self) -> int:
        return self.sequence_end - self.sequence_start

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


def slice_token_ppo_trajectory(
    trajectory: PackedTokenPPOTrajectory,
    start: int,
    end: int,
) -> PackedTokenPPOReplayBatch:
    """Slice a contiguous sequence microbatch without padding or repacking."""

    begin, stop = int(start), int(end)
    if not 0 <= begin < stop <= trajectory.batch_size:
        raise ValueError("sequence slice must be non-empty and in range")
    offsets = torch.cat([trajectory.lengths.new_zeros(1), trajectory.lengths.cumsum(dim=0)])
    return PackedTokenPPOReplayBatch(
        source=trajectory,
        sequence_start=begin,
        sequence_end=stop,
        token_start=int(offsets[begin].item()),
        token_end=int(offsets[stop].item()),
    )


@dataclass(frozen=True, slots=True)
class TokenPPOReplayResult:
    """Differentiable policy log-probabilities and critic values."""

    log_probs: torch.Tensor
    values: torch.Tensor
    sampling_temperature: float

    def __post_init__(self) -> None:
        for name, tensor in (("log_probs", self.log_probs), ("values", self.values)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or not tensor.is_floating_point():
                raise TypeError(f"{name} must be a floating tensor with shape [tokens]")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite")
        if self.log_probs.shape != self.values.shape:
            raise ValueError("replayed log_probs and values must have the same shape")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        object.__setattr__(self, "sampling_temperature", temperature)


@runtime_checkable
class TokenPPOReplayAdapter(Protocol):
    """Model-owned teacher-forced actor and critic replay."""

    module: object

    def replay(
        self,
        trajectory: PackedTokenPPOTrajectory | PackedTokenPPOReplayBatch,
        *,
        training: bool,
    ) -> TokenPPOReplayResult: ...


@runtime_checkable
class TokenPPORolloutAdapter(Protocol):
    """Behavior-policy sampler returning packed response tokens."""

    def rollout(
        self,
        request: TokenPPORolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> PackedTokenPPOTrajectory: ...


@runtime_checkable
class TokenPPOTerminalRewardAdapter(Protocol):
    """Score a packed trajectory into named per-sequence terminal rewards."""

    reward_ids: tuple[str, ...]

    def score(self, trajectory: PackedTokenPPOTrajectory) -> Mapping[str, torch.Tensor]: ...


__all__ = [
    "PackedTokenPPOReplayBatch",
    "PackedTokenPPOTrajectory",
    "TokenPPOReplayAdapter",
    "TokenPPOReplayResult",
    "TokenPPORolloutAdapter",
    "TokenPPORolloutRequest",
    "TokenPPOTerminalRewardAdapter",
    "slice_token_ppo_trajectory",
]
