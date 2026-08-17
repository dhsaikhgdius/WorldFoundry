"""Agentic iteration orchestration over the shared token-policy session."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..rl.algorithms.token_policy.session import (
    NativeTokenPolicyTrainingSession,
    TokenPolicyIterationResult,
)
from .contracts import AgenticRolloutRequest, AgenticTrajectory, agentic_trajectory_from_packed
from .rollout import AgenticRolloutAdapter


@dataclass(frozen=True, slots=True)
class AgenticIterationResult:
    """Environment trajectory together with learner updates and rewards."""

    trajectory: AgenticTrajectory
    token_policy: TokenPolicyIterationResult


class NativeAgenticTrainingSession:
    """Bridge multi-turn requests to the existing packed-token learner."""

    def __init__(
        self,
        rollout_adapter: AgenticRolloutAdapter,
        token_policy_session: NativeTokenPolicyTrainingSession,
    ) -> None:
        if not isinstance(rollout_adapter, AgenticRolloutAdapter):
            raise TypeError("rollout_adapter must implement AgenticRolloutAdapter")
        if not isinstance(token_policy_session, NativeTokenPolicyTrainingSession):
            raise TypeError("token_policy_session must be NativeTokenPolicyTrainingSession")
        if token_policy_session.rollout_adapter is not rollout_adapter:
            raise ValueError("token-policy session must use the same agentic rollout adapter")
        self.rollout_adapter = rollout_adapter
        self.token_policy_session = token_policy_session

    def wait_for_checkpoints(self) -> None:
        self.token_policy_session.wait_for_checkpoints()

    def train_iteration(
        self,
        request: AgenticRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> AgenticIterationResult:
        if not isinstance(request, AgenticRolloutRequest):
            raise TypeError("request must be AgenticRolloutRequest")
        token_result = self.token_policy_session.train_iteration(
            request.to_token_request(),
            generator=generator,
        )
        return AgenticIterationResult(
            trajectory=agentic_trajectory_from_packed(token_result.trajectory),
            token_policy=token_result,
        )


__all__ = ["AgenticIterationResult", "NativeAgenticTrainingSession"]
