"""Lazy public facade for native Cosmos cache preparation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CosmosTextFeatureEncoder": ".encoding",
    "CosmosVideoFeatureEncoder": ".encoding",
    "build_cosmos_video_decoding_dataset": ".training_cache",
    "materialize_cosmos_training_cache": ".training_cache",
    "prepare_cosmos_training_cache_from_audits": ".training_cache",
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
