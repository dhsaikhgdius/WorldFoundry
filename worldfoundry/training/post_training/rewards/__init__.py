"""Native reward contracts, adapters, scalarization, and HTTP execution."""

from __future__ import annotations

from importlib import import_module

from .contracts import RewardEvaluator, RewardRequest, RewardResult
from .scalarization import RewardScalarizationResult, WeightedRewardScalarizer

__all__ = [
    "RewardEvaluator",
    "RewardRequest",
    "RewardResult",
    "RewardScalarizationResult",
    "WeightedRewardScalarizer",
    "HTTPRewardEvaluator",
    "NativeRewardService",
    "RewardComponentOutput",
    "RewardComponentScorer",
    "RewardScorerRegistry",
    "WorkerGroupRewardScorer",
    "create_reward_service_app",
    "decode_wire_value",
    "encode_wire_value",
    "serve_reward_service",
    "AgenticCorrectnessConfig",
    "AgenticCorrectnessScorer",
    "AgenticToolSuccessConfig",
    "AgenticToolSuccessScorer",
    "AVRewardScorersConfig",
    "CLAP_SAMPLE_RATE",
    "CLAPConfig",
    "CLAPScorer",
    "VideoPickScoreConfig",
    "VideoPickScoreScorer",
    "build_av_reward_scorer_registry",
    "build_configured_reward_scorer_registry",
    "ScorerServiceConfig",
    "load_scorer_service_config",
]

_HTTP_EXPORTS = {
    "HTTPRewardEvaluator",
    "NativeRewardService",
    "RewardComponentOutput",
    "RewardComponentScorer",
    "RewardScorerRegistry",
    "WorkerGroupRewardScorer",
    "create_reward_service_app",
    "decode_wire_value",
    "encode_wire_value",
    "serve_reward_service",
}
_SCORER_EXPORTS = {
    "AgenticCorrectnessConfig",
    "AgenticCorrectnessScorer",
    "AgenticToolSuccessConfig",
    "AgenticToolSuccessScorer",
    "AVRewardScorersConfig",
    "CLAP_SAMPLE_RATE",
    "CLAPConfig",
    "CLAPScorer",
    "VideoPickScoreConfig",
    "VideoPickScoreScorer",
    "build_av_reward_scorer_registry",
    "build_configured_reward_scorer_registry",
    "ScorerServiceConfig",
    "load_scorer_service_config",
}


def __getattr__(name: str) -> object:
    if name in _HTTP_EXPORTS:
        module = import_module(".http", __name__)
    elif name in _SCORER_EXPORTS:
        module = import_module(".scorers", __name__)
    else:
        raise AttributeError(name)
    value = getattr(module, name)
    globals()[name] = value
    return value
