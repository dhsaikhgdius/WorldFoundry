"""Lazy public facade for native LTX cache preparation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LTXTextFeatureEncoder": ".encoding",
    "LTXVideoFeatureEncoder": ".encoding",
    "build_ltx_video_decoding_dataset": ".training_cache",
    "materialize_ltx_training_cache": ".training_cache",
    "materialize_ltx_rollout_conditioning_cache": ".rollout_cache",
    "prepare_ltx_training_cache_from_audits": ".training_cache",
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
