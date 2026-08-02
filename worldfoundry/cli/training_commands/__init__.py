"""Lightweight command registration for native training workflows."""

from __future__ import annotations

from importlib import import_module

_PUBLIC_EXPORTS = ("register_training_subparser",)
__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> object:
    if name != "register_training_subparser":
        raise AttributeError(name)
    value = getattr(import_module(".register", __name__), name)
    globals()[name] = value
    return value
