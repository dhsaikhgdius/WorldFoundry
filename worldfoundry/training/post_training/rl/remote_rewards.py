"""Trajectory reward adapters backed by HTTP or Ray scorer workers."""

from __future__ import annotations

from collections.abc import Mapping

from ..rewards.http.client import HTTPRewardEvaluator
from ..rewards.http.service import (
    NativeRewardService,
    RewardScorerRegistry,
    WorkerGroupRewardScorer,
)
from .trajectory_rewards import DecodedTerminalRewardAdapter


class HTTPTerminalRewardAdapter(DecodedTerminalRewardAdapter):
    """Score flow trajectories and DiffusionNFT terminals through HTTP."""

    def __init__(
        self,
        decoder: object,
        evaluator: HTTPRewardEvaluator,
        *,
        reward_ids: tuple[str, ...],
    ) -> None:
        self.http_evaluator = evaluator
        super().__init__(
            decoder,
            evaluator,
            reward_ids=reward_ids,
            evaluator_identity={"transport": "http", "base_url": evaluator.base_url},
        )


class WorkerGroupTerminalRewardAdapter(DecodedTerminalRewardAdapter):
    """Score decoded terminal videos with component-specific Ray workers."""

    def __init__(
        self,
        decoder: object,
        scorers: Mapping[str, WorkerGroupRewardScorer],
        *,
        fail_fast: bool = True,
    ) -> None:
        registry = RewardScorerRegistry()
        reward_ids = tuple(scorers)
        for reward_id, scorer in scorers.items():
            registry.register(reward_id, scorer)
        service = NativeRewardService(registry, fail_fast=fail_fast)
        self.scorers = dict(scorers)
        self.service = service
        super().__init__(
            decoder,
            service,
            reward_ids=reward_ids,
            evaluator_identity={"transport": "ray-worker-group"},
        )


__all__ = ["HTTPTerminalRewardAdapter", "WorkerGroupTerminalRewardAdapter"]
