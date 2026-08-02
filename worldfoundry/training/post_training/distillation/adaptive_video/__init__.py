"""Adaptive video distillation math and native execution."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ADAPTIVE_VIDEO_DATA_LOADER_STATE_SCHEMA": ".batching",
    "NativeAdaptiveVideoDataLoader": ".batching",
    "adaptive_video_real_batch_from_prepared": ".batching",
    "NativeAdaptiveVideoTrainingStack": ".builder",
    "build_native_adaptive_video_training_stack": ".builder",
    "AdaptiveVideoConfig": ".config",
    "AdaptiveVideoLossAdapter": ".contracts",
    "AdaptiveVideoRealBatch": ".contracts",
    "AdaptiveVideoTrainingBatch": ".contracts",
    "ADAPTIVE_VIDEO_ENGINE_STATE_SCHEMA": ".engine",
    "NativeAdaptiveVideoTrainEngine": ".engine",
    "AdaptiveRegressionObservation": ".math",
    "AdaptiveRegressionWeightResult": ".math",
    "TemporalRegularizationResult": ".math",
    "adaptive_regression_weights": ".math",
    "temporal_variance_regularization": ".math",
    "ADAPTIVE_VIDEO_OBJECTIVE_STATE_SCHEMA": ".objective",
    "AdaptiveVideoLossResult": ".objective",
    "FlowAdaptiveVideoLossAdapter": ".objective",
    "NativeAdaptiveVideoTrainingSession": ".session",
    "ADAPTIVE_REGRESSION_STATE_SCHEMA": ".state",
    "AdaptiveRegressionEMA": ".state",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
