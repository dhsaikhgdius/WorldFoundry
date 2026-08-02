"""Model-neutral contracts and infrastructure shared across post-training."""

from importlib import import_module

_EXPORTS = {
    "FlowPredictionAdapter": ".contracts",
    "DelayedModuleEMA": ".ema",
    "NativeClassifierFreeGuidance": ".prediction",
    "NativeFlowPredictionAdapter": ".prediction",
    "PostTrainingParallelContext": ".distributed",
    "ResolvedRoleCheckpoint": ".role_checkpoints",
    "TensorLike": ".contracts",
    "batch_shared_conditioning": ".batching",
    "resolve_role_checkpoint": ".role_checkpoints",
    "NativeSingleOptimizerTrainingSession": ".session",
    "SingleOptimizerEngine": ".session",
    "SingleOptimizerRunSummary": ".session",
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
