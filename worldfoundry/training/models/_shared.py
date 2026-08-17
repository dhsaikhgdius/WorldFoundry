"""Private helpers shared by model-family training adapters.

These were previously copy-pasted per adapter module and had started to
drift (signature differences, renamed keywords).  The module is deliberately
not re-exported through the ``models`` facade; adapter modules alias the
functions under their historical private names.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def component_module(component: object | None, *names: str) -> nn.Module | None:
    """Return the component itself or its first ``nn.Module`` attribute."""

    if isinstance(component, nn.Module):
        return component
    for name in names:
        value = getattr(component, name, None)
        if isinstance(value, nn.Module):
            return value
    return None


def module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Infer device and floating dtype from a module's first tensor."""

    reference = next(module.parameters(), None)
    if reference is None:
        reference = next(module.buffers(), None)
    if reference is None:
        return torch.device("cpu"), torch.float32
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    return reference.device, dtype


def freeze_module(module: nn.Module | None) -> None:
    """Disable gradients and switch to eval mode; ``None`` is a no-op."""

    if module is None:
        return
    module.requires_grad_(False)
    module.eval()


def merge_without_overwrite(
    destination: dict[str, object],
    source: Mapping[str, object],
    *,
    source_name: str,
    family: str,
) -> None:
    """Merge ``source`` into ``destination``, refusing to overwrite keys."""

    overlap = sorted(set(destination) & set(source))
    if overlap:
        raise ValueError(f"{source_name} collides with encoded {family} conditioning keys: {overlap}")
    destination.update(source)


__all__ = [
    "component_module",
    "freeze_module",
    "merge_without_overwrite",
    "module_device_dtype",
]
