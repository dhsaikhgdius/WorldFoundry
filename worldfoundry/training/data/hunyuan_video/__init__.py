"""Lazy public facade for HunyuanVideo rollout conditioning."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "HunyuanVideoTextFeatureEncoder": ".encoding",
    "materialize_hunyuan_video_rollout_conditioning_cache": ".rollout_cache",
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
