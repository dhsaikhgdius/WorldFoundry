"""Lazy public API for native classic token PPO."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeTokenPPODataLoader": ".batching",
    "NativeTokenPPOEngine": ".engine",
    "NativeTokenPPOTrainingRun": ".run",
    "NativeTokenPPOTrainingSession": ".session",
    "NativeTokenPPOTrainingStack": ".builder",
    "PackedTokenPPOReplayBatch": ".contracts",
    "PackedTokenPPOTrajectory": ".contracts",
    "SEQUENCE_MEAN_TOKEN_MEAN": ".objective",
    "SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED": ".objective",
    "TOKEN_MEAN": ".objective",
    "TOKEN_PPO_ENGINE_STATE_SCHEMA": ".engine",
    "TokenPPOAnchor": ".engine",
    "TokenPPOIterationResult": ".session",
    "TokenPPOLossTerms": ".objective",
    "TokenPPOReplayAdapter": ".contracts",
    "TokenPPOReplayResult": ".contracts",
    "TokenPPORolloutAdapter": ".contracts",
    "TokenPPORolloutRequest": ".contracts",
    "TokenPPORunSummary": ".run",
    "TokenPPOSample": ".batching",
    "TokenPPOStepResult": ".engine",
    "TokenPPOTerminalRewardAdapter": ".contracts",
    "build_native_token_ppo_training_stack": ".builder",
    "clipped_policy_losses": ".objective",
    "clipped_value_losses": ".objective",
    "materialize_token_ppo_training_run": ".run",
    "packed_gae": ".math",
    "scatter_terminal_rewards": ".math",
    "slice_token_ppo_trajectory": ".contracts",
    "token_ppo_loss": ".objective",
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
