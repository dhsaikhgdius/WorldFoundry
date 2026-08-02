"""Flow-DPPO policy objective."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FLOW_DPPO_ENGINE_STATE_SCHEMA": ".engine",
    "FlowDPPOIterationResult": ".session",
    "FlowDPPOLoss": ".objective",
    "FlowDPPOStepResult": ".engine",
    "FlowDPPOStageAlgorithm": ".algorithm",
    "NativeFlowDPPOEngine": ".engine",
    "NativeFlowDPPOTrainingSession": ".session",
    "build_native_flow_dppo_engine": ".engine",
    "flow_dppo_policy_loss": ".objective",
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
