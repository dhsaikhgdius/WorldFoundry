"""WorldFoundry-native DiffusionNFT forward-process policy training."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DiffusionNFTRewardAdapter": ".contracts",
    "DIFFUSION_NFT_ENGINE_STATE_SCHEMA": ".engine",
    "DIFFUSION_NFT_OLD_POLICY_SCHEDULES": ".contracts",
    "DiffusionNFTForwardSample": ".objective",
    "DiffusionNFTIterationResult": ".session",
    "DiffusionNFTLoss": ".objective",
    "DiffusionNFTRollout": ".contracts",
    "DiffusionNFTTerminalLatents": ".contracts",
    "DiffusionNFTRunSummary": ".session",
    "DiffusionNFTRewardWeights": ".objective",
    "DiffusionNFTStepResult": ".engine",
    "NativeDiffusionNFTEngine": ".engine",
    "NativeDiffusionNFTTerminalCollector": ".collection",
    "NativeDiffusionNFTTrainingStack": ".builder",
    "NativeDiffusionNFTTrainingSession": ".session",
    "OldPolicyRefresh": ".contracts",
    "build_native_diffusion_nft_training_stack": ".builder",
    "diffusion_nft_forward_process": ".objective",
    "diffusion_nft_loss": ".objective",
    "diffusion_nft_reward_weights": ".objective",
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
