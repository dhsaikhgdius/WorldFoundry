"""Lazy public API for process groups, meshes, and FSDP2."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FSDP2_APPLICATION_SCHEMA": ".fsdp",
    "FSDP2Application": ".fsdp",
    "apply_fsdp2": ".fsdp",
    "apply_fsdp2_frozen_reference": ".fsdp",
    "PARALLEL_PLAN_SCHEMA": ".parallel",
    "DistributedTrainingContext": ".parallel",
    "ParallelPlan": ".parallel",
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
