"""Packed variable-length contracts for autoregressive policy training."""

from __future__ import annotations

from collections import Counter
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


def _validated_ids(
    sample_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    samples = non_empty_ids(sample_ids, field_name="sample_ids", unique=True)
    groups = non_empty_ids(group_ids, field_name="group_ids", unique=False)
    if len(groups) != len(samples):
        raise ValueError("group_ids length must match sample_ids")
    incomplete = sorted(group for group, count in Counter(groups).items() if count < 2)
    if incomplete:
        raise ValueError(f"token-policy groups must contain at least two samples: {incomplete}")
    return samples, groups


@dataclass(frozen=True, slots=True)
class TokenRolloutRequest:
    """One grouped request consumed by a model-specific token rollout adapter."""

    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    policy_revision: str
    sampling_temperature: float = 1.0
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples, groups = _validated_ids(self.sample_ids, self.group_ids)
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "group_ids", groups)
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
class PackedTokenTrajectory:
    """Packed response tokens and the behavior-policy log-probability anchor.

    ``tokens`` and ``old_log_probs`` have shape ``[sum(lengths)]``. Individual
    sequence lengths may be zero; no padding tokens are inserted into the
    packed stream.
    """

    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    policy_revision: str
    tokens: torch.Tensor
    lengths: torch.Tensor
    old_log_probs: torch.Tensor
    sampling_temperature: float = 1.0
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples, groups = _validated_ids(self.sample_ids, self.group_ids)
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        if (
            not isinstance(self.lengths, torch.Tensor)
            or self.lengths.ndim != 1
            or int(self.lengths.shape[0]) != len(samples)
            or self.lengths.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError("lengths must be an integer tensor with shape [B]")
        if not bool((self.lengths >= 0).all()):
            raise ValueError("lengths must be non-negative")
        token_count = int(self.lengths.sum().item())
        if (
            not isinstance(self.tokens, torch.Tensor)
            or self.tokens.ndim != 1
            or int(self.tokens.shape[0]) != token_count
            or self.tokens.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError("tokens must be an integer tensor with shape [sum(lengths)]")
        if (
            not isinstance(self.old_log_probs, torch.Tensor)
            or self.old_log_probs.ndim != 1
            or int(self.old_log_probs.shape[0]) != token_count
            or not self.old_log_probs.is_floating_point()
        ):
            raise TypeError("old_log_probs must be a floating tensor with shape [sum(lengths)]")
        if not bool(torch.isfinite(self.old_log_probs).all()):
            raise ValueError("old_log_probs must be finite")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "group_ids", groups)
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


@dataclass(frozen=True, slots=True)
class PackedTokenReplayBatch:
    """A contiguous sequence view with the corresponding packed token span."""

    source: PackedTokenTrajectory
    sequence_start: int
    sequence_end: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, PackedTokenTrajectory):
            raise TypeError("source must be a PackedTokenTrajectory")
        values = (
            self.sequence_start,
            self.sequence_end,
            self.token_start,
            self.token_end,
        )
        if any(isinstance(value, bool) for value in values):
            raise TypeError("packed replay offsets must be integers")
        sequence_start, sequence_end, token_start, token_end = (int(value) for value in values)
        if not 0 <= sequence_start < sequence_end <= self.source.batch_size:
            raise ValueError("packed replay sequence interval must be non-empty")
        offsets = torch.cat(
            [
                self.source.lengths.new_zeros(1),
                self.source.lengths.cumsum(dim=0),
            ]
        )
        expected_start = int(offsets[sequence_start].item())
        expected_end = int(offsets[sequence_end].item())
        if (token_start, token_end) != (expected_start, expected_end):
            raise ValueError("packed replay token span differs from sequence lengths")
        object.__setattr__(self, "sequence_start", sequence_start)
        object.__setattr__(self, "sequence_end", sequence_end)
        object.__setattr__(self, "token_start", token_start)
        object.__setattr__(self, "token_end", token_end)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.source.sample_ids[self.sequence_start : self.sequence_end]

    @property
    def group_ids(self) -> tuple[str, ...]:
        return self.source.group_ids[self.sequence_start : self.sequence_end]

    @property
    def policy_revision(self) -> str:
        return self.source.policy_revision

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


def _slice_conditioning(
    values: Mapping[str, object],
    *,
    start: int,
    end: int,
    batch_size: int,
) -> Mapping[str, object]:
    def slice_value(value: object) -> object:
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                return value[start:end]
            return value
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
class TokenReplayResult:
    """Differentiable sampled-token log-probabilities from teacher forcing."""

    log_probs: torch.Tensor
    sampling_temperature: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.log_probs, torch.Tensor)
            or self.log_probs.ndim != 1
            or not self.log_probs.is_floating_point()
        ):
            raise TypeError("log_probs must be a floating tensor with shape [tokens]")
        if not bool(torch.isfinite(self.log_probs).all()):
            raise ValueError("replayed log_probs must be finite")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        object.__setattr__(self, "sampling_temperature", temperature)


@runtime_checkable
class TokenPolicyReplayAdapter(Protocol):
    """Model-owned teacher-forced replay used by the shared learner.

    The adapter returns sampled-response-token log-probabilities in exactly the
    packed order of ``trajectory.tokens``.  It owns model-specific attention,
    multimodal conditioning, and applies ``trajectory.sampling_temperature``
    to logits before selecting sampled-token log probabilities.  The returned
    temperature is checked by the shared engine.
    """

    module: object

    def replay(
        self,
        trajectory: PackedTokenTrajectory | PackedTokenReplayBatch,
        *,
        training: bool,
    ) -> TokenReplayResult: ...


@runtime_checkable
class TokenPolicyRolloutAdapter(Protocol):
    """Model-owned sampler returning only response tokens that participate in loss."""

    def rollout(
        self,
        request: TokenRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> PackedTokenTrajectory: ...


@runtime_checkable
class TokenTrajectoryRewardAdapter(Protocol):
    """Score one packed rollout into named per-sequence reward tensors."""

    reward_ids: tuple[str, ...]

    def score(
        self,
        trajectory: PackedTokenTrajectory,
    ) -> Mapping[str, torch.Tensor]: ...


__all__ = [
    "PackedTokenReplayBatch",
    "PackedTokenTrajectory",
    "TokenPolicyReplayAdapter",
    "TokenPolicyRolloutAdapter",
    "TokenReplayResult",
    "TokenRolloutRequest",
    "TokenTrajectoryRewardAdapter",
]
