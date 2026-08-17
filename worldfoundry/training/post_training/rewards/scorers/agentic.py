"""First-party rewards over completed Agentic transcripts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..contracts import RewardRequest
from ..http.service import RewardComponentOutput

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _normalized_answer(value: object) -> str:
    text = str(value).strip()
    match = _ANSWER_TAG.search(text)
    if match is not None:
        text = match.group(1)
    return " ".join(text.casefold().split())


def _invalid(message: str) -> RewardComponentOutput:
    return RewardComponentOutput(0.0, valid=False, diagnostics={"error": message})


@dataclass(frozen=True, slots=True)
class AgenticCorrectnessConfig:
    """Locate the expected answer in each request's conditions."""

    expected_answer_condition: str = "answer"

    def __post_init__(self) -> None:
        condition = self.expected_answer_condition.strip()
        if not condition:
            raise ValueError("expected_answer_condition must be non-empty")
        object.__setattr__(self, "expected_answer_condition", condition)


@dataclass(frozen=True, slots=True)
class AgenticToolSuccessConfig:
    """Declare the tool that a successful transcript must execute."""

    required_tool: str | None = None
    required_tool_condition: str | None = None

    def __post_init__(self) -> None:
        configured = tuple(value for value in (self.required_tool, self.required_tool_condition) if value is not None)
        if len(configured) != 1 or not configured[0].strip():
            raise ValueError("configure exactly one non-empty required_tool or required_tool_condition")
        if self.required_tool is not None:
            object.__setattr__(self, "required_tool", self.required_tool.strip())
        if self.required_tool_condition is not None:
            object.__setattr__(self, "required_tool_condition", self.required_tool_condition.strip())


class AgenticCorrectnessScorer:
    """Exact normalized terminal-answer reward with per-request validity."""

    def __init__(self, config: AgenticCorrectnessConfig | None = None) -> None:
        self.config = config or AgenticCorrectnessConfig()

    def score(self, requests: tuple[RewardRequest, ...]) -> tuple[RewardComponentOutput, ...]:
        outputs = []
        target_key = self.config.expected_answer_condition
        for request in requests:
            if target_key not in request.conditions:
                outputs.append(_invalid(f"conditions.{target_key} is required"))
                continue
            prediction = request.artifacts.get("prediction")
            if not isinstance(prediction, str):
                outputs.append(_invalid("artifacts.prediction must be text"))
                continue
            target = request.conditions[target_key]
            outputs.append(
                RewardComponentOutput(
                    float(_normalized_answer(prediction) == _normalized_answer(target)),
                )
            )
        return tuple(outputs)


def _required_tool(
    request: RewardRequest,
    config: AgenticToolSuccessConfig,
) -> str | None:
    if config.required_tool is not None:
        return config.required_tool
    condition = config.required_tool_condition
    if condition is None or condition not in request.conditions:
        return None
    value = request.conditions[condition]
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _tool_execution_succeeded(transcript: object, *, required_tool: str) -> bool | None:
    if not isinstance(transcript, Sequence) or isinstance(transcript, str | bytes | bytearray):
        return None
    calls: list[tuple[str, str]] = []
    results: dict[str, tuple[str, bool]] = {}
    for message in transcript:
        if not isinstance(message, Mapping):
            return None
        role = message.get("role")
        if role == "assistant":
            raw_calls = message.get("tool_calls", ())
            if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, str | bytes | bytearray):
                return None
            for call in raw_calls:
                if not isinstance(call, Mapping):
                    return None
                call_id = call.get("call_id")
                name = call.get("name")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    return None
                calls.append((call_id, name))
        elif role == "tool":
            call_id = message.get("tool_call_id")
            name = message.get("name")
            failed = message.get("tool_failed")
            if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(failed, bool):
                return None
            results[call_id] = (name, failed)

    if not calls:
        return False
    executed = tuple((name, results.get(call_id)) for call_id, name in calls)
    if any(result is None or result[0] != name for name, result in executed):
        return None
    return any(name == required_tool for _, name in calls) and all(
        result is not None and not result[1] for _, result in executed
    )


class AgenticToolSuccessScorer:
    """Reward successful execution of the explicitly required tool."""

    def __init__(self, config: AgenticToolSuccessConfig) -> None:
        self.config = config

    def score(self, requests: tuple[RewardRequest, ...]) -> tuple[RewardComponentOutput, ...]:
        outputs = []
        for request in requests:
            required_tool = _required_tool(request, self.config)
            if required_tool is None:
                field = self.config.required_tool_condition
                outputs.append(_invalid(f"conditions.{field} must name the required tool"))
                continue
            succeeded = _tool_execution_succeeded(
                request.artifacts.get("transcript"),
                required_tool=required_tool,
            )
            if succeeded is None:
                outputs.append(_invalid("artifacts.transcript has invalid tool-call metadata"))
                continue
            outputs.append(RewardComponentOutput(float(succeeded)))
        return tuple(outputs)


__all__ = [
    "AgenticCorrectnessConfig",
    "AgenticCorrectnessScorer",
    "AgenticToolSuccessConfig",
    "AgenticToolSuccessScorer",
]
