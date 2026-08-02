"""WorldFoundry-native AnyFlow pretraining and on-policy distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeAnyFlowBidirectionalAdapter": ".adapters",
    "NativeAnyFlowFARAdapter": ".adapters",
    "NativeAnyFlowScoreAdapter": ".adapters",
    "NativeAnyFlowBidirectionalOnPolicyLossAdapter": ".bidirectional_on_policy",
    "NativeAnyFlowBidirectionalPretrainLossAdapter": ".bidirectional_pretrain",
    "NativeAnyFlowOnPolicyTrainingStack": ".builder",
    "NativeAnyFlowPretrainingStack": ".builder",
    "build_native_anyflow_on_policy_training_stack": ".builder",
    "build_native_anyflow_pretraining_stack": ".builder",
    "AnyFlowBidirectionalOnPolicyConfig": ".config",
    "AnyFlowBidirectionalPretrainConfig": ".config",
    "AnyFlowFARConfig": ".config",
    "AnyFlowMapConfig": ".config",
    "AnyFlowOnPolicyConfig": ".config",
    "AnyFlowPretrainConfig": ".config",
    "AnyFlowBidirectionalAdapter": ".contracts",
    "AnyFlowFARAdapter": ".contracts",
    "AnyFlowLossResult": ".contracts",
    "AnyFlowOnPolicyLossAdapter": ".contracts",
    "AnyFlowPretrainLossAdapter": ".contracts",
    "AnyFlowScoreAdapter": ".contracts",
    "AnyFlowTrainingBatch": ".contracts",
    "AnyFlowEMA": ".ema",
    "ANYFLOW_ON_POLICY_ENGINE_STATE_SCHEMA": ".engine",
    "ANYFLOW_PRETRAIN_ENGINE_STATE_SCHEMA": ".engine",
    "AnyFlowOnPolicyResult": ".engine",
    "AnyFlowPretrainResult": ".engine",
    "NativeAnyFlowOnPolicyEngine": ".engine",
    "NativeAnyFlowPretrainEngine": ".engine",
    "NativeAnyFlowOnPolicyLossAdapter": ".on_policy",
    "NativeAnyFlowPretrainLossAdapter": ".pretrain",
    "AnyFlowRolloutChoice": ".rollout",
    "anyflow_bidirectional_rollout": ".rollout",
    "anyflow_rollout": ".rollout",
    "sample_rollout_choice": ".rollout",
    "AnyFlowOnPolicyRunSummary": ".session",
    "NativeAnyFlowOnPolicyTrainingSession": ".session",
    "NativeAnyFlowPretrainingSession": ".session",
    "AnyFlowDecisionRNG": ".synchronization",
    "AnyFlowTensorSynchronizer": ".synchronization",
    "ProcessGroupAnyFlowTensorSynchronizer": ".synchronization",
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
