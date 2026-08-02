"""Framework-owned canonical diffusion runners."""

from .autoregressive import AutoregressiveWindowRunner, WindowedDenoiser
from .base import DualConditionGuidanceRunner, NativeDiffusionRunner, RunnerComponents
from .chunked import ChunkedCacheDenoiser, ChunkedKVCacheRunner
from .multistage import (
    JointMultiStageDiffusionRunner,
    MultiModalLatentDecoder,
    MultiStageComponents,
    MultiStageLatentInitializer,
)
from .strategies import (
    DiffusionExecutor,
    ExecutionBuildContext,
    ExecutionStrategyRegistry,
    UnsupportedExecutionStrategyError,
    default_execution_strategy_registry,
)
from .staged import InferenceStage, InferenceStageGraph, InferenceStageRunner, StagedDiffusionPipeline
from .wan_staged import TeaCache, WanStagedPipeline, model_fn_wan_video

__all__ = [
    "DiffusionExecutor",
    "AutoregressiveWindowRunner",
    "DualConditionGuidanceRunner",
    "ChunkedCacheDenoiser",
    "ChunkedKVCacheRunner",
    "ExecutionBuildContext",
    "ExecutionStrategyRegistry",
    "JointMultiStageDiffusionRunner",
    "InferenceStage",
    "InferenceStageGraph",
    "InferenceStageRunner",
    "MultiModalLatentDecoder",
    "MultiStageComponents",
    "MultiStageLatentInitializer",
    "NativeDiffusionRunner",
    "RunnerComponents",
    "StagedDiffusionPipeline",
    "TeaCache",
    "WanStagedPipeline",
    "WindowedDenoiser",
    "UnsupportedExecutionStrategyError",
    "default_execution_strategy_registry",
    "model_fn_wan_video",
]
