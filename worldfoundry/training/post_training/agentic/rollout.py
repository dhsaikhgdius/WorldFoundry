"""Native multi-turn rollout over a model adapter and local tool environment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch

from ..rl.algorithms.token_policy.contracts import (
    PackedTokenTrajectory,
    TokenRolloutRequest,
)
from .contracts import (
    AgenticAssistantTurn,
    AgenticRolloutRequest,
    AgenticSampleRequest,
    AgenticSampleTrajectory,
    AgenticTrajectory,
    AgenticTurn,
    AgentMessage,
)
from .tools import AgentToolExecutor

AGENTIC_ROLLOUT_STATE_SCHEMA = "worldfoundry-agentic-rollout"


@runtime_checkable
class AgenticTurnModelAdapter(Protocol):
    """Model-owned generation of one assistant turn with sampled-token anchors."""

    def generate_turn(
        self,
        *,
        sample_id: str,
        messages: tuple[AgentMessage, ...],
        policy_revision: str,
        sampling_temperature: float,
        rollout_index: int,
        turn_index: int,
        conditioning: Mapping[str, object],
        generator: torch.Generator | None,
    ) -> AgenticAssistantTurn: ...


@runtime_checkable
class AgenticRolloutAdapter(Protocol):
    """Agentic rollout seam shared by local and remote implementations."""

    completed_rollouts: int

    def rollout(
        self,
        request: TokenRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> PackedTokenTrajectory: ...


class NativeAgenticRolloutAdapter:
    """Interact until model stop/length or the configured turn limit."""

    def __init__(
        self,
        model_adapter: AgenticTurnModelAdapter,
        tool_executor: AgentToolExecutor,
    ) -> None:
        if not isinstance(model_adapter, AgenticTurnModelAdapter):
            raise TypeError("model_adapter must implement AgenticTurnModelAdapter")
        if not isinstance(tool_executor, AgentToolExecutor):
            raise TypeError("tool_executor must implement AgentToolExecutor")
        self.model_adapter = model_adapter
        self.tool_executor = tool_executor
        self.completed_rollouts = 0

    def _rollout_sample(
        self,
        sample: AgenticSampleRequest,
        request: AgenticRolloutRequest,
        *,
        rollout_index: int,
        generator: torch.Generator | None,
    ) -> AgenticSampleTrajectory:
        messages = list(sample.messages)
        turns: list[AgenticTurn] = []
        terminal_reason = "turn_limit"
        for turn_index in range(request.max_turns):
            assistant = self.model_adapter.generate_turn(
                sample_id=sample.sample_id,
                messages=tuple(messages),
                policy_revision=request.policy_revision,
                sampling_temperature=request.sampling_temperature,
                rollout_index=rollout_index,
                turn_index=turn_index,
                conditioning=sample.conditioning,
                generator=generator,
            )
            if not isinstance(assistant, AgenticAssistantTurn):
                raise TypeError("model adapter must return AgenticAssistantTurn")
            messages.append(assistant.message)
            tool_results = tuple(self.tool_executor.execute(call) for call in assistant.message.tool_calls)
            messages.extend(tool_results)
            turns.append(AgenticTurn(assistant=assistant, tool_results=tool_results))
            if assistant.finish_reason != "tool_calls":
                terminal_reason = assistant.finish_reason
                break
        return AgenticSampleTrajectory(
            request=sample,
            turns=tuple(turns),
            terminal_reason=terminal_reason,
        )

    def rollout_sample(
        self,
        sample: AgenticSampleRequest,
        request: AgenticRolloutRequest,
        *,
        rollout_index: int,
        generator: torch.Generator | None = None,
    ) -> AgenticSampleTrajectory:
        """Run one sibling at a caller-owned rollout index."""

        return self._rollout_sample(
            sample,
            request,
            rollout_index=rollout_index,
            generator=generator,
        )

    def rollout_agentic(
        self,
        request: AgenticRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> AgenticTrajectory:
        if not isinstance(request, AgenticRolloutRequest):
            raise TypeError("request must be AgenticRolloutRequest")
        rollout_index = self.completed_rollouts
        completed: list[AgenticSampleTrajectory] = []
        for sample in request.samples:
            try:
                completed.append(
                    self._rollout_sample(
                        sample,
                        request,
                        rollout_index=rollout_index,
                        generator=generator,
                    )
                )
            except Exception:
                continue
        successful_counts = Counter(sample.request.group_id for sample in completed)
        samples = tuple(sample for sample in completed if successful_counts[sample.request.group_id] >= 2)
        selected_ids = {sample.request.sample_id for sample in samples}
        failed_sample_ids = tuple(
            sample.sample_id for sample in request.samples if sample.sample_id not in selected_ids
        )
        if not samples:
            raise RuntimeError("Agentic rollout produced no trainable sibling group")
        trajectory = AgenticTrajectory(
            samples=samples,
            policy_revision=request.policy_revision,
            sampling_temperature=request.sampling_temperature,
            rollout_index=rollout_index,
            failed_sample_ids=failed_sample_ids,
        )
        self.completed_rollouts += 1
        return trajectory

    def rollout(
        self,
        request: TokenRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> PackedTokenTrajectory:
        if not isinstance(request, TokenRolloutRequest):
            raise TypeError("request must be TokenRolloutRequest")
        agentic_request = request.conditioning.get("agentic_request")
        if not isinstance(agentic_request, AgenticRolloutRequest):
            raise ValueError("token rollout request does not contain an AgenticRolloutRequest")
        expected = agentic_request.to_token_request()
        if (
            request.sample_ids != expected.sample_ids
            or request.group_ids != expected.group_ids
            or request.policy_revision != expected.policy_revision
            or request.sampling_temperature != expected.sampling_temperature
        ):
            raise ValueError("token request differs from its agentic request")
        return self.rollout_agentic(agentic_request, generator=generator).to_packed_token_trajectory()

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": AGENTIC_ROLLOUT_STATE_SCHEMA,
            "completed_rollouts": self.completed_rollouts,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schema",
            "completed_rollouts",
        }:
            raise ValueError("agentic rollout state fields differ from the active schema")
        if state_dict["schema"] != AGENTIC_ROLLOUT_STATE_SCHEMA:
            raise ValueError("unsupported agentic rollout state schema")
        completed = state_dict["completed_rollouts"]
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("completed_rollouts must be non-negative")
        self.completed_rollouts = completed


__all__ = [
    "AGENTIC_ROLLOUT_STATE_SCHEMA",
    "AgenticRolloutAdapter",
    "AgenticTurnModelAdapter",
    "NativeAgenticRolloutAdapter",
]
