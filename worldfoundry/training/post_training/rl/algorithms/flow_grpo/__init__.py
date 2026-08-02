"""Flow-GRPO objective, engine factory, and typed session."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ClippedPolicyLoss": ".objective",
    "FLOW_GRPO_ENGINE_STATE_SCHEMA": ".engine",
    "FlowGRPOIterationResult": ".session",
    "FlowGRPOStepResult": ".engine",
    "FlowGRPOStageAlgorithm": ".algorithm",
    "NativeFlowGRPOEngine": ".engine",
    "NativeFlowGRPOTrainingSession": ".session",
    "build_native_flow_grpo_engine": ".engine",
    "clipped_policy_loss": ".objective",
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
