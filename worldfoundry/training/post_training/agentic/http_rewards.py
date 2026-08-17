"""HTTP service rewards over completed agentic trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from worldfoundry.training.post_training.rewards.contracts import (
    RewardEvaluator,
    RewardRequest,
    RewardResult,
)
from worldfoundry.training.post_training.rewards.http.client import HTTPRewardEvaluator
from worldfoundry.training.post_training.rl.algorithms.token_policy.contracts import (
    PackedTokenTrajectory,
    TokenTrajectoryRewards,
)

from .contracts import (
    AgenticSampleTrajectory,
    AgentMessage,
    agentic_trajectory_from_packed,
)


def _message_payload(message: AgentMessage) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": dict(call.arguments),
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        payload.update(
            {
                "tool_call_id": message.tool_call_id,
                "name": message.name,
                "tool_failed": message.tool_failed,
            }
        )
    return payload


def _question(sample: AgenticSampleTrajectory) -> str:
    user_text = [
        message.content for message in sample.request.messages if message.role == "user" and message.content.strip()
    ]
    if user_text:
        return "\n".join(user_text)
    return "\n".join(message.content for message in sample.request.messages if message.content.strip())


class HTTPAgenticRewardAdapter:
    """Batch terminal questions, answers, and transcripts through HTTP."""

    def __init__(
        self,
        evaluator: HTTPRewardEvaluator,
        *,
        reward_ids: Sequence[str],
    ) -> None:
        if not isinstance(evaluator, RewardEvaluator):
            raise TypeError("evaluator must implement RewardEvaluator")
        resolved = tuple(str(value) for value in reward_ids)
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("HTTP Agentic reward ids must be non-empty and unique")
        self.evaluator = evaluator
        self.reward_ids = resolved

    def _request(
        self,
        sample: AgenticSampleTrajectory,
        *,
        rollout_id: str,
    ) -> RewardRequest:
        terminal = sample.turns[-1].assistant.message.content
        return RewardRequest(
            request_id=sample.request.sample_id,
            rollout_id=rollout_id,
            prompt=_question(sample),
            conditions=sample.request.conditioning,
            artifacts={
                "question": _question(sample),
                "prediction": terminal,
                "transcript": [_message_payload(message) for message in sample.messages],
                "terminal_reason": sample.terminal_reason,
            },
            reward_ids=self.reward_ids,
            metadata={
                "group_id": sample.request.group_id,
                "turn_count": len(sample.turns),
            },
        )

    def score(self, trajectory: PackedTokenTrajectory) -> TokenTrajectoryRewards:
        agentic = agentic_trajectory_from_packed(trajectory)
        rollout_id = f"agentic-rollout-{agentic.rollout_index:08d}"
        requests = tuple(self._request(sample, rollout_id=rollout_id) for sample in agentic.samples)
        results = self.evaluator.evaluate(requests)
        by_id = {result.request_id: result for result in results}
        if len(by_id) != len(requests) or set(by_id) != {request.request_id for request in requests}:
            raise ValueError("HTTP reward results differ from the Agentic request batch")
        ordered: tuple[RewardResult, ...] = tuple(by_id[request.request_id] for request in requests)
        if any(result.rollout_id != rollout_id or set(result.values) != set(self.reward_ids) for result in ordered):
            raise ValueError("HTTP reward result fields differ from the Agentic request")
        device = trajectory.old_log_probs.device
        return TokenTrajectoryRewards(
            values={
                reward_id: torch.tensor(
                    [result.values[reward_id] for result in ordered],
                    device=device,
                    dtype=torch.float32,
                )
                for reward_id in self.reward_ids
            },
            valid={
                reward_id: torch.tensor(
                    [result.valid[reward_id] for result in ordered],
                    device=device,
                    dtype=torch.bool,
                )
                for reward_id in self.reward_ids
            },
        )


__all__ = ["HTTPAgenticRewardAdapter"]
