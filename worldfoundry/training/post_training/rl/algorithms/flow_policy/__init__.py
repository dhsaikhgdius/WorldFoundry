"""Lazy exports for the shared flow-policy learner runtime."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FlowPolicyAlgorithmRuntime": ".runtime",
    "FlowPolicyIterationResult": ".session",
    "FlowPolicyStepResult": ".engine",
    "NativeFlowPolicyEngine": ".engine",
    "NativeFlowPolicyTrainingSession": ".session",
    "NativeFlowPolicyTrainingStack": ".builder",
    "build_native_flow_policy_training_stack": ".builder",
    "resolve_flow_policy_algorithm_runtime": ".runtime",
    "shared_variance_gaussian_kl": ".reference_kl",
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
