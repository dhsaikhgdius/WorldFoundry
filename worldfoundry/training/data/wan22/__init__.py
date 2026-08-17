"""Lazy public facade for Wan2.2 rollout conditioning."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "WAN22_T2V_A14B_REPOSITORY": ".assets",
    "WAN22_TOKENIZER_FILES": ".assets",
    "Wan22TextCheckpoints": ".assets",
    "wan22_text_checkpoints": ".assets",
    "WAN22_T2V_A14B_MODEL": ".rollout_cache",
    "materialize_wan22_rollout_conditioning_cache": ".rollout_cache",
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
