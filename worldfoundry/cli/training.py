"""Lazy compatibility facade for WorldFoundry training CLI commands."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "register_training_subparser": ".training_commands.register",
}

_PUBLIC_EXPORTS = ("register_training_subparser",)
__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
