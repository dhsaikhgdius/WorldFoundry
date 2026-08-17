"""Native AnyFlow model-family execution."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ANYFLOW_DATA_LOADER_STATE_SCHEMA": ".data",
    "AnyFlowCachedDataLoader": ".data",
    "anyflow_batch_from_cached": ".data",
    "materialize_anyflow_training_run": ".materialize",
    "AnyFlowRoleBundle": ".roles",
    "AnyFlowTrainableRoles": ".roles",
    "AnyFlowTrainingRun": ".run",
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
