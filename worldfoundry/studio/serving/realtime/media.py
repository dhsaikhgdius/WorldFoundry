"""Frame normalization, transport encoding, and bounded realtime queues."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image

from worldfoundry.core.acceleration.frame_prefetch import prefetch_to_numpy
from worldfoundry.studio.execution import _normalize_frame_list, _to_uint8_rgb

MIN_OUTPUT_WIDTH = 160
MIN_OUTPUT_HEIGHT = 90
MAX_OUTPUT_WIDTH = 1920
MAX_OUTPUT_HEIGHT = 1920
MAX_OUTPUT_PIXELS = 1920 * 1080


class FrameQueuePolicy(str, Enum):
    """Congestion behavior for decoded realtime frames."""

    LATEST_INTERACTIVE = "latest-interactive"
    ORDERED_QUALITY = "ordered-quality"

    @classmethod
    def from_value(cls, value: "FrameQueuePolicy | str") -> "FrameQueuePolicy":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().lower().replace("_", "-")
        aliases = {
            "latest": cls.LATEST_INTERACTIVE,
            "interactive": cls.LATEST_INTERACTIVE,
            "ordered": cls.ORDERED_QUALITY,
            "quality": cls.ORDERED_QUALITY,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown frame queue policy {value!r}; expected one of: {supported}.") from exc


class LatestFrameBuffer:
    """Bounded frame queue with selectable latency/quality congestion policy.

    The historical name remains as a compatibility surface. In
    ``latest-interactive`` mode it briefly backpressures and then evicts stale
    frames. In ``ordered-quality`` mode it waits for capacity and never drops a
    generated frame.
    """

    def __init__(
        self,
        *,
        maxsize: int,
        backpressure_ms: int = 0,
        policy: FrameQueuePolicy | str = FrameQueuePolicy.LATEST_INTERACTIVE,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self.backpressure_s = max(float(backpressure_ms), 0.0) / 1000.0
        self.policy = FrameQueuePolicy.from_value(policy)
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=maxsize)
        self._closed_event = asyncio.Event()
        self.dropped_frames = 0
        self.last_enqueue_ms = 0.0
        self.closed = False

    def qsize(self) -> int:
        return self._queue.qsize()

    async def _put_ordered(self, frame: np.ndarray) -> bool:
        while not self.closed:
            try:
                self._queue.put_nowait(frame)
                return True
            except asyncio.QueueFull:
                try:
                    await asyncio.wait_for(self._closed_event.wait(), timeout=0.05)
                except TimeoutError:
                    continue
        return False

    async def put_chunk(self, frames: list[np.ndarray]) -> int:
        if self.closed:
            return 0
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self.backpressure_s
        accepted = 0
        for frame in frames:
            rgb = np.ascontiguousarray(frame)
            if self.policy is FrameQueuePolicy.ORDERED_QUALITY:
                if not await self._put_ordered(rgb):
                    break
                accepted += 1
                continue
            if self._queue.full() and self.backpressure_s > 0:
                remaining = deadline - loop.time()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(self._queue.put(rgb), timeout=remaining)
                        accepted += 1
                        continue
                    except TimeoutError:
                        pass
            while self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped_frames += 1
                except asyncio.QueueEmpty:
                    break
            if self.closed:
                break
            self._queue.put_nowait(rgb)
            accepted += 1
        self.last_enqueue_ms = (loop.time() - started) * 1000.0
        return accepted

    async def get(self) -> np.ndarray:
        frame = await self._queue.get()
        if frame is None:
            raise EOFError("frame buffer closed")
        return frame

    def get_nowait(self) -> np.ndarray:
        frame = self._queue.get_nowait()
        if frame is None:
            raise EOFError("frame buffer closed")
        return frame

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._closed_event.set()
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(None)


def realtime_frames_from_result(result: Any) -> list[np.ndarray]:
    """Extract in-memory RGB frames without invoking artifact exporters."""

    if isinstance(result, Iterator):
        close = getattr(result, "close", None)
        try:
            result = list(result)
        finally:
            if callable(close):
                close()

    candidates: list[Any] = []
    if isinstance(result, Mapping):
        for key in ("sr_videos", "videos", "frames", "video", "output", "images"):
            if key in result:
                candidates.append(result[key])
    else:
        candidates.append(result)

    for candidate in candidates:
        if isinstance(candidate, Image.Image):
            return [_to_uint8_rgb(candidate)]
        if isinstance(candidate, (list, tuple)):
            # Start every eligible device-to-host copy before waiting on the
            # first frame. Plain NumPy/PIL/tensor lists remain a no-op here.
            for frame in candidate:
                prefetch_to_numpy(frame)
            materializers = [getattr(frame, "to_numpy", None) for frame in candidate]
            if candidate and all(callable(materialize) for materialize in materializers):
                return [_to_uint8_rgb(materialize()) for materialize in materializers if callable(materialize)]
        frames = _normalize_frame_list(candidate)
        if frames:
            return [np.ascontiguousarray(frame) for frame in frames]
    raise RuntimeError(
        "Realtime stream returned no in-memory RGB frames. The model integration "
        "must return a tensor/array/image chunk instead of only an artifact path."
    )


def resize_rgb_frame(
    frame: np.ndarray,
    output_resolution: tuple[int, int] | None,
) -> np.ndarray:
    """Prepare one transport frame without ever enlarging model output."""

    rgb = np.ascontiguousarray(frame)
    if output_resolution is None or (rgb.shape[1], rgb.shape[0]) == output_resolution:
        return rgb
    width, height = output_resolution
    source_height, source_width = rgb.shape[:2]
    if width > MAX_OUTPUT_WIDTH or height > MAX_OUTPUT_HEIGHT or width * height > MAX_OUTPUT_PIXELS:
        raise ValueError(f"realtime transport resolution {width}x{height} exceeds its safe output budget.")
    if width > source_width or height > source_height:
        raise ValueError(
            f"realtime transport cannot upscale {source_width}x{source_height} model output to {width}x{height}."
        )
    scale = min(width / source_width, height / source_height)
    fitted = (
        max(min(int(round(source_width * scale)), width), 1),
        max(min(int(round(source_height * scale)), height), 1),
    )
    resized = Image.fromarray(rgb, mode="RGB").resize(
        fitted,
        resample=Image.Resampling.BILINEAR,
    )
    if fitted == output_resolution:
        return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))
    canvas = Image.new("RGB", output_resolution, "black")
    canvas.paste(resized, ((width - fitted[0]) // 2, (height - fitted[1]) // 2))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def resize_rgb_frames(
    frames: list[np.ndarray],
    *,
    output_resolution: tuple[int, int] | None,
) -> list[np.ndarray]:
    return [resize_rgb_frame(frame, output_resolution) for frame in frames]


def encode_jpeg_frames(
    frames: list[np.ndarray],
    *,
    quality: int,
    subsampling: int = 1,
    output_resolution: tuple[int, int] | None = None,
) -> list[bytes]:
    """Encode a generated chunk for the same-port WebSocket fallback."""

    packets: list[bytes] = []
    for frame in frames:
        output = io.BytesIO()
        image = Image.fromarray(
            resize_rgb_frame(frame, output_resolution),
            mode="RGB",
        )
        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=False,
            subsampling=subsampling,
        )
        packets.append(output.getvalue())
    return packets


__all__ = [
    "FrameQueuePolicy",
    "LatestFrameBuffer",
    "MAX_OUTPUT_HEIGHT",
    "MAX_OUTPUT_PIXELS",
    "MAX_OUTPUT_WIDTH",
    "MIN_OUTPUT_HEIGHT",
    "MIN_OUTPUT_WIDTH",
    "encode_jpeg_frames",
    "realtime_frames_from_result",
    "resize_rgb_frame",
    "resize_rgb_frames",
]
