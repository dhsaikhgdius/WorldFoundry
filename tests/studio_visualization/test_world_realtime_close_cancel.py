"""RT-01: close_active must not hang behind a blocked max_workers=1 executor."""

from __future__ import annotations

import asyncio
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

realtime_backend = pytest.importorskip(
    "worldfoundry.studio.visualization.backends.world_realtime"
)


class _BlockingRuntime:
    """Minimal runtime stand-in that deadlocks reset until abandoned."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rt01-test")
        self._cancel = threading.Event()
        self._block = threading.Event()
        self._started = threading.Event()
        self.abandoned = False
        # Occupy the single worker so reset would hang without abandon.
        self._executor.submit(self._block.wait)

    def request_cancel(self) -> None:
        self._cancel.set()

    def clear_cancel(self) -> None:
        self._cancel.clear()

    def _abandon_executor(self) -> None:
        self.abandoned = True
        old = self._executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rt01-test")
        self._block.set()
        old.shutdown(wait=False, cancel_futures=True)

    async def reset(self) -> None:
        self._started.set()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, time.sleep, 60.0)


def test_close_active_abandons_blocked_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_CLOSE_TIMEOUT_SECONDS", "0.2")

    async def scenario() -> None:
        runtime = _BlockingRuntime()
        manager = realtime_backend.RealtimePeerManager(
            runtime=runtime,
            fps=8,
            chunk_frames=4,
        )
        peer = types.SimpleNamespace(close=lambda: asyncio.sleep(0))
        frames = types.SimpleNamespace(close=lambda: None)
        active = realtime_backend._ActivePeer(
            peer=peer,
            channel=None,
            frames=frames,
            resampler=types.SimpleNamespace(),
        )
        manager._active = active

        started = time.perf_counter()
        await manager.close_active()
        elapsed = time.perf_counter() - started

        assert runtime.abandoned is True
        assert elapsed < 5.0
        assert manager.active is False

    asyncio.run(scenario())


def test_close_timeout_helper_defaults() -> None:
    assert realtime_backend._close_timeout_s() == 30.0
