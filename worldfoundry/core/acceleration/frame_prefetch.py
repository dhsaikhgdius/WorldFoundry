# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-to-host frame prefetch primitives for realtime presentation.

Adapted from NVIDIA FlashDreams.  Torch is imported lazily so CPU-only Studio
and manifest tooling do not acquire an accelerator dependency.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import numpy.typing as npt

_STREAMS_LOCK = threading.Lock()
_HOST_COPY_STREAMS: dict[int, Any] = {}


class CudaHostPrefetch:
    """Stage one CUDA tensor into pinned host memory on a reusable side stream."""

    def __init__(self, tensor: Any, *, source_event: Any | None = None) -> None:
        self._tensor = tensor
        self._source_event = source_event
        self._host_tensor: Any | None = None
        self._done_event: Any | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> bool:
        if self._started:
            return self._host_tensor is not None
        self._started = True
        try:
            import torch
        except ImportError:
            return False
        if not torch.is_tensor(self._tensor) or not self._tensor.is_cuda:
            return False
        try:
            host_tensor = torch.empty(
                tuple(self._tensor.shape),
                dtype=self._tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            copy_stream = _host_copy_stream(torch, self._tensor.device)
            with torch.cuda.device(self._tensor.device):
                if self._source_event is not None:
                    copy_stream.wait_event(self._source_event)
                with torch.cuda.stream(copy_stream):
                    host_tensor.copy_(self._tensor, non_blocking=True)
                    self._tensor.record_stream(copy_stream)
                    done_event = torch.cuda.Event()
                    done_event.record(copy_stream)
        except Exception:
            return False
        self._host_tensor = host_tensor
        self._done_event = done_event
        return True

    def to_numpy(self) -> np.ndarray:
        if self._host_tensor is None:
            raise RuntimeError("CUDA host prefetch was not started successfully.")
        if self._done_event is not None:
            self._done_event.synchronize()
        return np.ascontiguousarray(self._host_tensor.numpy())


class LazyCudaFrame:
    """Keep a decoded frame on CUDA until transport requests host materialization."""

    def __init__(
        self,
        frames_hwc_uint8: Any,
        frame_index: int,
        *,
        source_event: object | None = None,
    ) -> None:
        self._frames_hwc_uint8: Any | None = frames_hwc_uint8
        self._frame_index = int(frame_index)
        self._source_event = source_event
        self._host: np.ndarray | None = None
        self._prefetch: CudaHostPrefetch | None = None

    def prefetch_to_numpy(self) -> None:
        if self._host is not None or self._prefetch is not None or self._frames_hwc_uint8 is None:
            return
        frame = self._frames_hwc_uint8[self._frame_index].detach()
        prefetch = CudaHostPrefetch(frame, source_event=self._source_event)
        if prefetch.start():
            self._prefetch = prefetch

    def to_numpy(self) -> np.ndarray:
        if self._host is not None:
            return self._host
        if self._prefetch is not None:
            self._host = self._prefetch.to_numpy()
        else:
            if self._frames_hwc_uint8 is None:
                raise RuntimeError("Lazy CUDA frame lost its source before materialization.")
            synchronize = getattr(self._source_event, "synchronize", None)
            if callable(synchronize):
                synchronize()
            self._host = np.ascontiguousarray(self._frames_hwc_uint8[self._frame_index].detach().cpu().numpy())
        self._frames_hwc_uint8 = None
        self._prefetch = None
        return self._host

    def to_cuda_tensor(self) -> Any:
        if self._frames_hwc_uint8 is None:
            raise RuntimeError("Lazy CUDA frame was already materialized on the host.")
        return self._frames_hwc_uint8[self._frame_index]

    def to_cuda_event(self) -> object | None:
        return self._source_event if self._frames_hwc_uint8 is not None else None

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        array = self.to_numpy()
        if dtype is not None:
            target_dtype = np.dtype(dtype)
            if copy is False and target_dtype != array.dtype:
                raise ValueError("Cannot honor copy=False while converting the frame dtype.")
            array = array.astype(target_dtype, copy=False)
        return np.array(array, copy=True) if copy is True else array


def prefetch_to_numpy(frame: object) -> None:
    """Start host materialization when a frame exposes the lazy protocol."""

    prefetch = getattr(frame, "prefetch_to_numpy", None)
    if callable(prefetch):
        prefetch()


def _host_copy_stream(torch: Any, device: Any) -> Any:
    index = getattr(device, "index", None)
    key = 0 if index is None else int(index)
    with _STREAMS_LOCK:
        stream = _HOST_COPY_STREAMS.get(key)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
            _HOST_COPY_STREAMS[key] = stream
        return stream


__all__ = ["CudaHostPrefetch", "LazyCudaFrame", "prefetch_to_numpy"]
