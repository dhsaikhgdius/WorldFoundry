"""Lazy public API for native Reward-Forcing distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeRewardForcingDataLoader": ".batching",
    "NativeRewardForcingTrainingStack": ".builder",
    "build_native_reward_forcing_training_stack": ".builder",
    "RewardForcingConfig": ".config",
    "RewardForcingAlgorithmSpec": ("worldfoundry.training.recipes.post_training.algorithms.reward_forcing"),
    "parse_reward_forcing_algorithm": ("worldfoundry.training.recipes.post_training.algorithms.reward_forcing"),
    "MotionQualityRewardAdapter": ".contracts",
    "RewardForcingCausalAdapter": ".contracts",
    "RewardForcingDecoderAdapter": ".contracts",
    "RewardForcingTrainingBatch": ".contracts",
    "NativeRewardForcingTrainEngine": ".engine",
    "RewardedProxyLoss": ".math",
    "reward_forcing_multiplier": ".math",
    "rewarded_dmd_proxy_loss": ".math",
    "NativeRewardForcingLossAdapter": ".objective",
    "VideoAlignMotionQualityReward": ".reward",
    "NativeRewardForcingTrainingSession": ".session",
    "WanRewardForcingChunkAdapter": ".wan",
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
