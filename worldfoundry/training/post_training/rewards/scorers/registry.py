"""Construction of native reward scorer registries."""

from __future__ import annotations

from ..http.service import RewardScorerRegistry
from .agentic import AgenticCorrectnessScorer, AgenticToolSuccessScorer
from .clap import CLAPScorer
from .config import AVRewardScorersConfig
from .service_config import ScorerServiceConfig
from .video_pickscore import VideoPickScoreScorer


def build_av_reward_scorer_registry(
    config: AVRewardScorersConfig | None = None,
) -> RewardScorerRegistry:
    """Build the lazy scorer registry used by an audio-video HTTP sidecar."""

    resolved = config or AVRewardScorersConfig()
    registry = RewardScorerRegistry()
    registry.register("videopickscore", VideoPickScoreScorer(resolved.videopickscore))
    registry.register("clap", CLAPScorer(resolved.clap))
    return registry


def build_configured_reward_scorer_registry(
    config: ScorerServiceConfig,
) -> RewardScorerRegistry:
    """Build only the reward components selected for one HTTP service process."""

    registry = RewardScorerRegistry()
    if config.videopickscore is not None:
        registry.register("videopickscore", VideoPickScoreScorer(config.videopickscore))
    if config.clap is not None:
        registry.register("clap", CLAPScorer(config.clap))
    if config.correctness is not None:
        registry.register("correctness", AgenticCorrectnessScorer(config.correctness))
    if config.tool_success is not None:
        registry.register("tool-success", AgenticToolSuccessScorer(config.tool_success))
    return registry


__all__ = ["build_av_reward_scorer_registry", "build_configured_reward_scorer_registry"]
