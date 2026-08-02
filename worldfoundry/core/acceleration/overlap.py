# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small execution runners for optional realtime work overlap.

Adapted from NVIDIA FlashDreams with explicit error propagation and a common
CPU/CUDA lifecycle surface.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


class SynchronousOverlap:
    """Execute submitted work immediately behind the overlap-runner API."""

    def __init__(self, *, name: str = "sync-overlap") -> None:
        self.name = name
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return False

    @property
    def last_error(self) -> BaseException | None:
        return self._error

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        del name
        self._error = None
        try:
            result = work()
            del result
        except BaseException as exc:
            self._error = exc
            raise

    def wait(self, *, timeout_s: float | None = None, raise_error: bool = False) -> bool:
        del timeout_s
        if raise_error and self._error is not None:
            raise self._error
        return True

    def close(self, *, wait: bool = True) -> None:
        del wait


class HostThreadOverlap:
    """Execute at most one deferred callback on a daemon host thread."""

    def __init__(self, *, name: str = "host-overlap", daemon: bool = True) -> None:
        self.name = name
        self._daemon = daemon
        self._done = threading.Event()
        self._done.set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return not self._done.is_set()

    @property
    def last_error(self) -> BaseException | None:
        return self._error

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        with self._lock:
            if self.pending:
                raise RuntimeError(f"{self.name} already has pending overlap work.")
            self._error = None
            self._done.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(work,),
                name=name or self.name,
                daemon=self._daemon,
            )
            self._thread.start()

    def wait(self, *, timeout_s: float | None = None, raise_error: bool = False) -> bool:
        if timeout_s is not None and timeout_s < 0.0:
            raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")
        completed = self._done.wait(timeout=timeout_s)
        if completed and raise_error and self._error is not None:
            raise self._error
        return completed

    def close(self, *, wait: bool = True) -> None:
        thread = self._thread
        if wait and thread is not None:
            thread.join()

    def _run(self, work: Callable[[], Any]) -> None:
        try:
            result = work()
            del result
        except BaseException as exc:
            self._error = exc
        finally:
            self._done.set()


class CudaStreamOverlap:
    """Enqueue work on a side CUDA stream and publish a completion event."""

    def __init__(
        self,
        *,
        name: str = "cuda-stream-overlap",
        device: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.name = name
        self._torch = torch_module if torch_module is not None else importlib.import_module("torch")
        if not self._torch.cuda.is_available():
            raise RuntimeError("CUDA stream overlap requires an available CUDA device.")
        self._device = self._torch.device(device) if device is not None else None
        self._stream = self._torch.cuda.Stream(device=self._device)
        self._event: Any | None = None
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return self._event is not None and not self._event.query()

    @property
    def last_error(self) -> BaseException | None:
        return self._error

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        del name
        if not self.wait(timeout_s=0.0):
            raise RuntimeError(f"{self.name} already has pending overlap work.")
        self._error = None
        try:
            device_context = nullcontext() if self._device is None else self._torch.cuda.device(self._device)
            with device_context, self._torch.cuda.stream(self._stream):
                result = work()
                del result
                event = self._torch.cuda.Event()
                event.record(self._stream)
        except BaseException as exc:
            self._error = exc
            raise
        self._event = event

    def wait(self, *, timeout_s: float | None = None, raise_error: bool = False) -> bool:
        event = self._event
        if event is None:
            completed = True
        elif timeout_s is None:
            event.synchronize()
            completed = True
        else:
            completed = _wait_for_cuda_event(event, timeout_s=timeout_s)
        if completed and raise_error and self._error is not None:
            raise self._error
        return completed

    def close(self, *, wait: bool = True) -> None:
        if wait:
            self.wait(raise_error=True)


def _wait_for_cuda_event(event: Any, *, timeout_s: float) -> bool:
    if timeout_s < 0.0:
        raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")
    deadline = time.monotonic() + timeout_s
    while not event.query():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.001)
    return True


__all__ = ["CudaStreamOverlap", "HostThreadOverlap", "SynchronousOverlap"]
