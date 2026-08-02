"""Lazy public API for native autoregressive policy training."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MAX_SEQUENCE_LOG_RATIO": ".objectives",
    "NativeTokenPolicyEngine": ".engine",
    "NativeTokenPolicyTrainingStack": ".builder",
    "NativeTokenPolicyTrainingSession": ".session",
    "PackedTokenReplayBatch": ".contracts",
    "PackedTokenTrajectory": ".contracts",
    "SEQUENCE_MEAN_TOKEN_MEAN": ".reduction",
    "SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED": ".reduction",
    "TOKEN_MEAN": ".reduction",
    "TOKEN_POLICY_ENGINE_STATE_SCHEMA": ".engine",
    "TokenCPPOStage": ".stages",
    "TokenDPPOStage": ".stages",
    "TokenDRPOStage": ".stages",
    "TokenGRPOStage": ".stages",
    "TokenGSPOStage": ".stages",
    "TokenObjective": ".objectives",
    "TokenPolicyIterationResult": ".session",
    "TokenPolicyReplayAdapter": ".contracts",
    "TokenPolicyRolloutAdapter": ".contracts",
    "TokenPolicyStage": ".stages",
    "TokenPolicyStageLoss": ".stages",
    "TokenPolicyStepResult": ".engine",
    "TokenReplayResult": ".contracts",
    "TokenRolloutRequest": ".contracts",
    "TokenTrajectoryRewardAdapter": ".contracts",
    "build_token_policy_stage": ".runtime",
    "build_native_token_policy_training_stack": ".builder",
    "expand_sequence_values": ".packing",
    "packed_token_offsets": ".packing",
    "slice_packed_token_trajectory": ".packing",
    "token_cppo_objective": ".objectives",
    "token_dppo_objective": ".objectives",
    "token_drpo_objective": ".objectives",
    "token_grpo_objective": ".objectives",
    "token_gspo_objective": ".objectives",
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
