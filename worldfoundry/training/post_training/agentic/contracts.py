"""Functional contracts for multi-turn tool-use policy rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal

import torch

from ..rl.algorithms.token_policy.contracts import (
    PackedTokenReplayBatch,
    PackedTokenTrajectory,
    TokenRolloutRequest,
)

AgentRole = Literal["system", "user", "assistant", "tool"]
AgentFinishReason = Literal["stop", "tool_calls", "length"]

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _text(value: object, *, field_name: str) -> str:
    resolved = str(value)
    if not resolved.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return resolved


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """One structured tool request emitted by the policy."""

    call_id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        object.__setattr__(self, "call_id", _text(self.call_id, field_name="call_id"))
        object.__setattr__(self, "name", _text(self.name, field_name="tool name"))
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType({str(key): value for key, value in self.arguments.items()}),
        )


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """A conversation message, including assistant calls and tool results."""

    role: AgentRole
    content: str = ""
    tool_calls: tuple[AgentToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    tool_failed: bool = False

    def __post_init__(self) -> None:
        role = str(self.role)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported agent message role: {role!r}")
        calls = tuple(self.tool_calls)
        if not all(isinstance(call, AgentToolCall) for call in calls):
            raise TypeError("tool_calls must contain AgentToolCall values")
        if role == "assistant":
            if self.tool_call_id is not None or self.name is not None or self.tool_failed:
                raise ValueError("assistant messages cannot contain tool-result fields")
        elif role == "tool":
            if calls:
                raise ValueError("tool-result messages cannot emit tool calls")
            object.__setattr__(self, "tool_call_id", _text(self.tool_call_id, field_name="tool_call_id"))
            object.__setattr__(self, "name", _text(self.name, field_name="tool name"))
        elif calls or self.tool_call_id is not None or self.name is not None or self.tool_failed:
            raise ValueError("system and user messages cannot contain tool fields")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", str(self.content))
        object.__setattr__(self, "tool_calls", calls)


@dataclass(frozen=True, slots=True)
class AgenticAssistantTurn:
    """An assistant message and the sampled tokens that produced it."""

    message: AgentMessage
    token_ids: torch.Tensor
    old_log_probs: torch.Tensor
    finish_reason: AgentFinishReason

    def __post_init__(self) -> None:
        if not isinstance(self.message, AgentMessage) or self.message.role != "assistant":
            raise TypeError("assistant turn message must have role='assistant'")
        if (
            not isinstance(self.token_ids, torch.Tensor)
            or self.token_ids.ndim != 1
            or self.token_ids.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError("assistant token_ids must be a one-dimensional integer tensor")
        if (
            not isinstance(self.old_log_probs, torch.Tensor)
            or self.old_log_probs.ndim != 1
            or self.old_log_probs.shape != self.token_ids.shape
            or not self.old_log_probs.is_floating_point()
        ):
            raise TypeError("assistant old_log_probs must be a floating tensor matching token_ids")
        if not bool(torch.isfinite(self.old_log_probs).all()):
            raise ValueError("assistant old_log_probs must be finite")
        reason = str(self.finish_reason)
        if reason not in {"stop", "tool_calls", "length"}:
            raise ValueError(f"unsupported assistant finish_reason: {reason!r}")
        if (reason == "tool_calls") != bool(self.message.tool_calls):
            raise ValueError("tool_calls finish reason must exactly match emitted calls")
        object.__setattr__(self, "finish_reason", reason)


@dataclass(frozen=True, slots=True)
class AgenticTurn:
    """One policy turn and the environment messages produced by its calls."""

    assistant: AgenticAssistantTurn
    tool_results: tuple[AgentMessage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.assistant, AgenticAssistantTurn):
            raise TypeError("assistant must be AgenticAssistantTurn")
        results = tuple(self.tool_results)
        if not all(isinstance(message, AgentMessage) and message.role == "tool" for message in results):
            raise TypeError("tool_results must contain role='tool' messages")
        call_ids = tuple(call.call_id for call in self.assistant.message.tool_calls)
        result_ids = tuple(message.tool_call_id for message in results)
        if result_ids != call_ids:
            raise ValueError("tool results must match assistant tool calls in order")
        object.__setattr__(self, "tool_results", results)


@dataclass(frozen=True, slots=True)
class AgenticSampleRequest:
    """Initial conversation and model inputs for one grouped sample."""

    sample_id: str
    group_id: str
    messages: tuple[AgentMessage, ...]
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not messages or not all(isinstance(message, AgentMessage) for message in messages):
            raise ValueError("agentic sample messages must be a non-empty AgentMessage tuple")
        if any(message.role in {"assistant", "tool"} for message in messages):
            raise ValueError("initial agentic messages may only contain system and user roles")
        if not isinstance(self.conditioning, Mapping):
            raise TypeError("agentic sample conditioning must be a mapping")
        object.__setattr__(self, "sample_id", _text(self.sample_id, field_name="sample_id"))
        object.__setattr__(self, "group_id", _text(self.group_id, field_name="group_id"))
        object.__setattr__(self, "messages", messages)
        object.__setattr__(
            self,
            "conditioning",
            MappingProxyType({str(key): value for key, value in self.conditioning.items()}),
        )


@dataclass(frozen=True, slots=True)
class AgenticRolloutRequest:
    """A grouped batch for multi-turn agent-environment interaction."""

    samples: tuple[AgenticSampleRequest, ...]
    policy_revision: str
    sampling_temperature: float = 1.0
    max_turns: int = 8

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples or not all(isinstance(sample, AgenticSampleRequest) for sample in samples):
            raise ValueError("samples must be a non-empty AgenticSampleRequest tuple")
        sample_ids = tuple(sample.sample_id for sample in samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("agentic sample_ids must be unique")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        if isinstance(self.max_turns, bool) or int(self.max_turns) <= 0:
            raise ValueError("max_turns must be a positive integer")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "policy_revision", _text(self.policy_revision, field_name="policy_revision"))
        object.__setattr__(self, "sampling_temperature", temperature)
        object.__setattr__(self, "max_turns", int(self.max_turns))

    def to_token_request(self) -> TokenRolloutRequest:
        return TokenRolloutRequest(
            sample_ids=tuple(sample.sample_id for sample in self.samples),
            group_ids=tuple(sample.group_id for sample in self.samples),
            policy_revision=self.policy_revision,
            sampling_temperature=self.sampling_temperature,
            conditioning={"agentic_request": self},
        )


@dataclass(frozen=True, slots=True)
class AgenticSampleTrajectory:
    """Complete environment-visible trajectory for one policy sample."""

    request: AgenticSampleRequest
    turns: tuple[AgenticTurn, ...]
    terminal_reason: Literal["stop", "length", "turn_limit"]

    def __post_init__(self) -> None:
        if not isinstance(self.request, AgenticSampleRequest):
            raise TypeError("request must be AgenticSampleRequest")
        turns = tuple(self.turns)
        if not turns or not all(isinstance(turn, AgenticTurn) for turn in turns):
            raise ValueError("agentic trajectory must contain at least one turn")
        reason = str(self.terminal_reason)
        if reason not in {"stop", "length", "turn_limit"}:
            raise ValueError(f"unsupported terminal_reason: {reason!r}")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "terminal_reason", reason)

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        transcript = list(self.request.messages)
        for turn in self.turns:
            transcript.append(turn.assistant.message)
            transcript.extend(turn.tool_results)
        return tuple(transcript)

    @property
    def token_ids(self) -> torch.Tensor:
        return torch.cat(tuple(turn.assistant.token_ids for turn in self.turns))

    @property
    def old_log_probs(self) -> torch.Tensor:
        return torch.cat(tuple(turn.assistant.old_log_probs for turn in self.turns))


@dataclass(frozen=True, slots=True)
class AgenticTrajectory:
    """Batch trajectory with conversion into the shared token-policy learner."""

    samples: tuple[AgenticSampleTrajectory, ...]
    policy_revision: str
    sampling_temperature: float
    rollout_index: int
    failed_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples or not all(isinstance(sample, AgenticSampleTrajectory) for sample in samples):
            raise ValueError("agentic trajectory samples must be non-empty")
        temperature = float(self.sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        if isinstance(self.rollout_index, bool) or int(self.rollout_index) < 0:
            raise ValueError("rollout_index must be non-negative")
        failed = tuple(str(value) for value in self.failed_sample_ids)
        sample_ids = {sample.request.sample_id for sample in samples}
        if len(set(failed)) != len(failed) or any(not value.strip() for value in failed) or set(failed) & sample_ids:
            raise ValueError("failed_sample_ids must be unique, non-empty, and disjoint")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "policy_revision", _text(self.policy_revision, field_name="policy_revision"))
        object.__setattr__(self, "sampling_temperature", temperature)
        object.__setattr__(self, "rollout_index", int(self.rollout_index))
        object.__setattr__(self, "failed_sample_ids", failed)

    def to_packed_token_trajectory(self) -> PackedTokenTrajectory:
        token_chunks = tuple(sample.token_ids for sample in self.samples)
        log_prob_chunks = tuple(sample.old_log_probs for sample in self.samples)
        lengths = torch.tensor(
            [int(tokens.numel()) for tokens in token_chunks],
            dtype=torch.int64,
            device=token_chunks[0].device,
        )
        return PackedTokenTrajectory(
            sample_ids=tuple(sample.request.sample_id for sample in self.samples),
            group_ids=tuple(sample.request.group_id for sample in self.samples),
            policy_revision=self.policy_revision,
            tokens=torch.cat(token_chunks),
            lengths=lengths,
            old_log_probs=torch.cat(log_prob_chunks),
            sampling_temperature=self.sampling_temperature,
            conditioning={
                "agentic_samples": self.samples,
                "agentic_rollout_index": self.rollout_index,
            },
            excluded_sample_ids=self.failed_sample_ids,
        )


def agentic_trajectory_from_packed(
    trajectory: PackedTokenTrajectory | PackedTokenReplayBatch,
) -> AgenticTrajectory:
    """Recover the environment trajectory attached by the agentic rollout adapter."""

    samples = trajectory.conditioning.get("agentic_samples")
    rollout_index = trajectory.conditioning.get("agentic_rollout_index")
    if not isinstance(samples, tuple) or not all(isinstance(sample, AgenticSampleTrajectory) for sample in samples):
        raise ValueError("packed trajectory does not contain agentic samples")
    if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
        raise ValueError("packed trajectory does not contain an agentic rollout index")
    value = AgenticTrajectory(
        samples=samples,
        policy_revision=trajectory.policy_revision,
        sampling_temperature=trajectory.sampling_temperature,
        rollout_index=rollout_index,
        failed_sample_ids=trajectory.excluded_sample_ids,
    )
    if tuple(sample.request.sample_id for sample in samples) != trajectory.sample_ids:
        raise ValueError("packed and agentic sample order differs")
    return value


__all__ = [
    "AgentFinishReason",
    "AgentMessage",
    "AgentRole",
    "AgentToolCall",
    "AgenticAssistantTurn",
    "AgenticRolloutRequest",
    "AgenticSampleRequest",
    "AgenticSampleTrajectory",
    "AgenticTrajectory",
    "AgenticTurn",
    "agentic_trajectory_from_packed",
]
