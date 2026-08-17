"""Direct state comparisons used by training round-trip gates."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import torch


def snapshot_state(value: object) -> object:
    """Copy nested runtime state while moving tensors to owned CPU storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): snapshot_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot_state(item) for item in value)
    if isinstance(value, list):
        return [snapshot_state(item) for item in value]
    return copy.deepcopy(value)


def assert_state_equal(expected: object, actual: object, *, path: str = "state") -> int:
    """Assert exact equality recursively and return the compared leaf count."""

    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path} type differs")
        if expected.dtype != actual.dtype or tuple(expected.shape) != tuple(actual.shape):
            raise AssertionError(f"{path} tensor metadata differs")
        if not torch.equal(expected.cpu(), actual.detach().cpu()):
            raise AssertionError(f"{path} tensor values differ")
        return 1
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise AssertionError(f"{path} type differs")
        if set(expected) != set(actual):
            raise AssertionError(f"{path} keys differ")
        return sum(
            assert_state_equal(expected[key], actual[key], path=f"{path}[{key!r}]")
            for key in expected
        )
    if isinstance(expected, (tuple, list)) or isinstance(actual, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f"{path} sequence differs")
        return sum(
            assert_state_equal(left, right, path=f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        )
    if expected != actual:
        raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")
    return 1


def state_changed(before: object, after: object) -> bool:
    """Return whether two nested states differ."""

    try:
        assert_state_equal(before, after)
    except AssertionError:
        return True
    return False


def file_size_inventory(path: str | Path) -> dict[str, int]:
    """List artifact payload files and their byte sizes."""

    root = Path(path)
    if root.is_file():
        return {root.name: root.stat().st_size}
    return {
        candidate.relative_to(root).as_posix(): candidate.stat().st_size
        for candidate in sorted(root.rglob("*"))
        if candidate.is_file()
    }


__all__ = [
    "assert_state_equal",
    "file_size_inventory",
    "snapshot_state",
    "state_changed",
]
