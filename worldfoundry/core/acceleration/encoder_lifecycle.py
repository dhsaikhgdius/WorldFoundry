# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle helpers for one-shot encoder and initialization stages.

Adapted from NVIDIA FlashDreams so model integrations share the same reference
release, garbage collection, and CUDA allocator cleanup behavior.
"""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


def setup_one_shot_encoder(
    config: Any,
    *,
    device: Any | Callable[[], Any] | None = None,
    torch_module: Any | None = None,
) -> Any:
    """Instantiate an encoder config and move torch modules to ``device``."""

    encoder = config.setup()
    if device is None:
        return encoder
    torch = torch_module if torch_module is not None else _maybe_import_torch()
    module_cls = getattr(getattr(torch, "nn", None), "Module", None)
    if module_cls is not None and isinstance(encoder, module_cls):
        encoder = encoder.to(device=device() if callable(device) else device)
    return encoder


def ensure_one_shot_encoder(
    encoder: Any | None,
    config: Any | None,
    *,
    device: Any | Callable[[], Any] | None = None,
    name: str = "encoder",
    required: bool = True,
    torch_module: Any | None = None,
) -> Any | None:
    """Return an existing encoder or lazily reconstruct it from ``config``."""

    if encoder is not None:
        return encoder
    if config is None:
        if required:
            raise RuntimeError(f"{name} is unloaded and has no configuration for reloading.")
        return None
    return setup_one_shot_encoder(config, device=device, torch_module=torch_module)


def release_one_shot_encoder_references(
    owner: Any,
    *attrs: str,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> tuple[str, ...]:
    """Clear encoder attributes and release allocator blocks they retained."""

    released: list[str] = []
    for attr in attrs:
        if getattr(owner, attr, None) is not None:
            released.append(attr)
        setattr(owner, attr, None)
    collect_and_release_cuda_memory(
        device=device,
        synchronize_cuda=synchronize_cuda,
        empty_cuda_cache=empty_cuda_cache,
        torch_module=torch_module,
    )
    return tuple(released)


def offload_module_to_cpu(
    module: Any,
    *,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> Any:
    """Move a reusable module to CPU and release its no-longer-used CUDA blocks."""

    moved = module.cpu()
    collect_and_release_cuda_memory(
        device=device,
        synchronize_cuda=synchronize_cuda,
        empty_cuda_cache=empty_cuda_cache,
        torch_module=torch_module,
    )
    return moved


def collect_and_release_cuda_memory(
    *,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> None:
    """Collect unreachable objects and optionally trim the CUDA allocator."""

    gc.collect()
    if not empty_cuda_cache and not synchronize_cuda:
        return
    torch = torch_module if torch_module is not None else _maybe_import_torch()
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if cuda is None or not callable(is_available) or not is_available():
        return
    if synchronize_cuda:
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize() if device is None else synchronize(device)
    if empty_cuda_cache:
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def move_tensors_to_cpu(value: Any, *, torch_module: Any | None = None) -> Any:
    """Recursively detach tensor containers from accelerator memory."""

    torch = torch_module if torch_module is not None else _maybe_import_torch()
    is_tensor = getattr(torch, "is_tensor", None)
    if callable(is_tensor) and is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {key: move_tensors_to_cpu(item, torch_module=torch) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_cpu(item, torch_module=torch) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_cpu(item, torch_module=torch) for item in value)
    return value


def run_one_shot_encoder_stage(
    stage: Callable[[], Any],
    *,
    release: Callable[[], Any] | None = None,
    cpu_result: bool = True,
    torch_module: Any | None = None,
) -> Any:
    """Run an encoder-only stage without gradients and always release it."""

    torch = torch_module if torch_module is not None else _maybe_import_torch()
    no_grad = getattr(torch, "no_grad", None)
    context = no_grad() if callable(no_grad) else nullcontext()
    try:
        with context:
            result = stage()
        return move_tensors_to_cpu(result, torch_module=torch) if cpu_result else result
    finally:
        if release is not None:
            released = release()
            del released


def _maybe_import_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


__all__ = [
    "collect_and_release_cuda_memory",
    "ensure_one_shot_encoder",
    "move_tensors_to_cpu",
    "offload_module_to_cpu",
    "release_one_shot_encoder_references",
    "run_one_shot_encoder_stage",
    "setup_one_shot_encoder",
]
