"""Small validation primitives shared by post-training runtimes."""

from __future__ import annotations

from math import isfinite


def positive_float(value: float, *, field_name: str) -> float:
    """Resolve one finite, strictly positive floating-point value."""

    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def non_negative_int(value: object, *, field_name: str) -> int:
    """Resolve one non-negative integer while rejecting booleans."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def validate_stateful_or_none(value: object | None, *, field_name: str) -> None:
    """Require optional state to implement the checkpoint state protocol."""

    if value is not None and (
        not callable(getattr(value, "state_dict", None)) or not callable(getattr(value, "load_state_dict", None))
    ):
        raise TypeError(f"{field_name} must expose state_dict/load_state_dict")


__all__ = ["non_negative_int", "positive_float", "validate_stateful_or_none"]
