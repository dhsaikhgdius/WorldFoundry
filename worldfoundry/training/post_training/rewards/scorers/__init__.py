"""Native media-model and Agentic transcript reward scorers.

Importing this package does not load Transformers, TorchVision, or TorchAudio.
Media-model weights and optional dependencies are loaded on first scoring use.
"""

from .agentic import (
    AgenticCorrectnessConfig,
    AgenticCorrectnessScorer,
    AgenticToolSuccessConfig,
    AgenticToolSuccessScorer,
)
from .clap import CLAP_SAMPLE_RATE, CLAPScorer
from .config import AVRewardScorersConfig, CLAPConfig, VideoPickScoreConfig
from .registry import build_av_reward_scorer_registry, build_configured_reward_scorer_registry
from .service_config import ScorerServiceConfig, load_scorer_service_config
from .video_pickscore import VideoPickScoreScorer

__all__ = [
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
