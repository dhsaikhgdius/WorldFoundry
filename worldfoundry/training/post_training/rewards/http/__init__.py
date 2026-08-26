"""HTTP transport for WorldFoundry reward evaluators."""

from .client import HTTPRewardEvaluator
from .codec import decode_wire_value, encode_wire_value
from .service import (
    REWARD_SERVICE_TOKEN_ENV,
    NativeRewardService,
    RewardComponentOutput,
    RewardComponentScorer,
    RewardScorerRegistry,
    WorkerGroupRewardScorer,
    configured_reward_service_token,
    create_reward_service_app,
    require_reward_auth_token_for_host,
    serve_reward_service,
)

__all__ = [
    "REWARD_SERVICE_TOKEN_ENV",
    "HTTPRewardEvaluator",
    "NativeRewardService",
    "RewardComponentOutput",
    "RewardComponentScorer",
    "RewardScorerRegistry",
    "WorkerGroupRewardScorer",
    "configured_reward_service_token",
    "create_reward_service_app",
    "decode_wire_value",
    "encode_wire_value",
    "require_reward_auth_token_for_host",
    "serve_reward_service",
]
