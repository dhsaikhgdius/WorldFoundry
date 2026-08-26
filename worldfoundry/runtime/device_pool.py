"""Import-light CUDA device discovery and worker grouping helpers.

``CudaDeviceLeasePool`` provides thread-safe in-process allocation and, by
default, cross-process exclusivity via ``fcntl.flock`` files under
``${WORLDFOUNDRY_HOME}/locks/gpu-<token>.lock``. Set
``WORLDFOUNDRY_GPU_CROSS_PROCESS_LOCK=0`` to disable the file locks (process-
local Condition only). A FIFO ticket queue prevents large acquires from being
starved by a stream of smaller ones.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable

_DISABLED_DEVICE_VALUES = frozenset({"", "-1", "none", "void"})
_ALL_DEVICE_VALUES = frozenset({"all"})
_FALSEY = frozenset({"0", "false", "no", "off"})
_TOKEN_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _worldfoundry_home() -> Path:
    raw = str(os.environ.get("WORLDFOUNDRY_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "worldfoundry"


def _cross_process_lock_enabled() -> bool:
    raw = str(os.environ.get("WORLDFOUNDRY_GPU_CROSS_PROCESS_LOCK", "1")).strip().lower()
    return raw not in _FALSEY


def _lock_path_for_token(token: str, *, locks_dir: Path | None = None) -> Path:
    safe = _TOKEN_SAFE.sub("_", str(token)).strip("._") or "device"
    root = locks_dir if locks_dir is not None else _worldfoundry_home() / "locks"
    return root / f"gpu-{safe}.lock"


def _try_acquire_flock(
    token: str, *, locks_dir: Path | None = None
) -> tuple[bool, IO[str] | None]:
    """Attempt a non-blocking exclusive flock for ``token``.

    Returns ``(True, handle)`` on success, ``(True, None)`` when locking is
    unavailable on this platform (proceed without a file lock), and
    ``(False, None)`` when another process holds the lock.
    """

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platforms
        return True, None
    path = _lock_path_for_token(token, locks_dir=locks_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False, None
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
    except OSError:
        pass
    return True, handle


def _release_flock(handle: IO[str] | None) -> None:
    if handle is None:
        return
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    try:
        handle.close()
    except OSError:
        pass


@dataclass
class CudaDeviceLease:
    """A reversible reservation of one or more physical CUDA device tokens."""

    _pool: "CudaDeviceLeasePool" = field(repr=False)
    tokens: tuple[str, ...]
    _released: bool = field(default=False, init=False, repr=False)
    _flock_handles: tuple[IO[str] | None, ...] = field(default=(), repr=False)

    @property
    def visible_devices(self) -> str:
        """Return the value suitable for ``CUDA_VISIBLE_DEVICES``."""

        return ",".join(self.tokens)

    @property
    def allocation_waiting(self) -> bool:
        """Return whether another worker is waiting for devices from this pool."""

        return self._pool.waiting_count > 0

    def release(self) -> None:
        """Return the reserved devices to the pool; repeated calls are harmless."""

        if self._released:
            return
        self._released = True
        for handle in self._flock_handles:
            _release_flock(handle)
        self._flock_handles = ()
        self._pool.release(self.tokens)

    def __enter__(self) -> "CudaDeviceLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class CudaDeviceLeasePool:
    """Thread-safe allocator for non-overlapping CUDA worker assignments.

    Within one process, a ``threading.Condition`` serializes acquires. Across
    processes that share ``WORLDFOUNDRY_HOME``, per-device ``flock`` files
    provide exclusivity. Acquires are FIFO-ticketed so a large request waiting
    at the head is not starved by smaller later requests.
    """

    def __init__(
        self,
        devices: Sequence[str],
        *,
        locks_dir: Path | str | None = None,
        cross_process: bool | None = None,
    ) -> None:
        normalized = normalize_cuda_device_groups(tuple(str(device) for device in devices))
        self._devices = tuple(normalized)
        self._available = list(self._devices)
        self._leased: set[str] = set()
        self._waiting = 0
        self._condition = threading.Condition()
        self._ticket_queue: deque[object] = deque()
        self._locks_dir = Path(locks_dir).expanduser() if locks_dir else None
        self._cross_process = _cross_process_lock_enabled() if cross_process is None else bool(cross_process)

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    @property
    def available_count(self) -> int:
        with self._condition:
            return len(self._available)

    @property
    def waiting_count(self) -> int:
        with self._condition:
            return self._waiting

    def _claim_with_flocks(
        self,
        requested: int,
    ) -> tuple[tuple[str, ...], tuple[IO[str] | None, ...]] | None:
        """Try to reserve ``requested`` in-process devices and flock them.

        Must be called while holding ``self._condition``.
        """

        if len(self._available) < requested:
            return None
        claimed: list[str] = []
        handles: list[IO[str] | None] = []
        remaining = list(self._available)
        for token in list(remaining):
            if len(claimed) >= requested:
                break
            handle: IO[str] | None = None
            if self._cross_process:
                ok, handle = _try_acquire_flock(token, locks_dir=self._locks_dir)
                if not ok:
                    # Held by another process — leave in available for later.
                    continue
            claimed.append(token)
            handles.append(handle)
            remaining.remove(token)
        if len(claimed) < requested:
            for handle in handles:
                _release_flock(handle)
            return None
        self._available = remaining
        self._leased.update(claimed)
        return tuple(claimed), tuple(handles)

    def acquire(
        self,
        count: int = 1,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        poll_interval: float = 0.1,
    ) -> CudaDeviceLease:
        """Wait for and reserve ``count`` devices, honoring cooperative cancellation."""

        requested = max(int(count), 1)
        if requested > len(self._devices):
            raise RuntimeError(
                f"requested {requested} CUDA devices, but only {len(self._devices)} are available"
            )
        ticket = object()
        with self._condition:
            self._ticket_queue.append(ticket)
            self._waiting += 1
            try:
                while True:
                    if cancel_requested is not None and cancel_requested():
                        raise RuntimeError("CUDA device allocation cancelled")
                    # FIFO: only the head may claim, so large waits are not starved.
                    if self._ticket_queue and self._ticket_queue[0] is not ticket:
                        self._condition.wait(timeout=max(float(poll_interval), 0.01))
                        continue
                    claimed = self._claim_with_flocks(requested)
                    if claimed is not None:
                        tokens, handles = claimed
                        self._ticket_queue.popleft()
                        self._condition.notify_all()
                        return CudaDeviceLease(self, tokens, _flock_handles=handles)
                    self._condition.wait(timeout=max(float(poll_interval), 0.01))
            finally:
                self._waiting -= 1
                try:
                    self._ticket_queue.remove(ticket)
                except ValueError:
                    pass
                self._condition.notify_all()

    def release(self, tokens: Sequence[str]) -> None:
        """Release previously leased tokens and wake waiting allocators."""

        with self._condition:
            released = {str(token) for token in tokens if str(token) in self._leased}
            if not released:
                return
            self._leased.difference_update(released)
            self._available = [device for device in self._devices if device not in self._leased]
            self._condition.notify_all()


def cuda_device_tokens(value: object) -> tuple[str, ...]:
    """Normalize a comma-separated CUDA visibility value into device tokens."""
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    tokens = tuple(
        token.strip()
        for item in raw_items
        for token in item.split(",")
        if token.strip()
    )
    if len(tokens) == 1 and tokens[0].lower() in _DISABLED_DEVICE_VALUES:
        return ()
    return tokens


def normalize_cuda_device_groups(values: Sequence[str] | None) -> tuple[str, ...]:
    """Validate non-overlapping worker groups used as CUDA_VISIBLE_DEVICES values."""
    groups: list[str] = []
    assigned: set[str] = set()
    raw_groups: Sequence[str] = (values,) if isinstance(values, str) else (values or ())
    for raw_group in raw_groups:
        tokens = cuda_device_tokens(raw_group)
        if not tokens:
            raise ValueError("CUDA worker device groups cannot be empty")
        lowered = {token.lower() for token in tokens}
        if lowered & _ALL_DEVICE_VALUES:
            raise ValueError("CUDA worker device groups must list concrete device ids or UUIDs, not 'all'")
        if lowered & _DISABLED_DEVICE_VALUES:
            raise ValueError("CUDA worker device groups must list enabled device ids or UUIDs")
        if len(lowered) != len(tokens):
            raise ValueError(f"CUDA worker device group contains duplicates: {raw_group!r}")
        overlap = assigned.intersection(lowered)
        if overlap:
            duplicate = ", ".join(sorted(overlap))
            raise ValueError(f"CUDA devices cannot be assigned to multiple model workers: {duplicate}")
        assigned.update(lowered)
        groups.append(",".join(tokens))
    return tuple(groups)


def discover_cuda_device_tokens(
    environ: Mapping[str, str] | None = None,
    *,
    timeout_seconds: float = 3.0,
) -> tuple[str, ...]:
    """Discover visible NVIDIA device ids without importing or initializing Torch."""
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" in env:
        configured = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip()
        if configured.lower() not in _ALL_DEVICE_VALUES:
            return cuda_device_tokens(configured)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=max(float(timeout_seconds), 0.1),
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            completed = None
        if completed is not None and completed.returncode == 0:
            devices = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
            if devices:
                # NVIDIA_VISIBLE_DEVICES is interpreted by the container runtime and may
                # contain host indices that are remapped to different local CUDA indices.
                return devices

    container_devices = str(env.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
    if container_devices and container_devices.lower() not in _ALL_DEVICE_VALUES:
        return cuda_device_tokens(container_devices)
    return ()


def cuda_device_discovery_source(environ: Mapping[str, str] | None = None) -> str:
    """Describe whether concrete CUDA visibility comes from the environment."""
    env = os.environ if environ is None else environ
    configured = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if configured and configured.lower() not in _ALL_DEVICE_VALUES:
        return "environment"
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    container_devices = str(env.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
    if container_devices and container_devices.lower() not in _ALL_DEVICE_VALUES:
        return "environment"
    return "unavailable"


def default_cuda_device_groups(
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return one non-overlapping model-worker group per visible CUDA device."""
    return normalize_cuda_device_groups(discover_cuda_device_tokens(environ))


__all__ = [
    "CudaDeviceLease",
    "CudaDeviceLeasePool",
    "cuda_device_discovery_source",
    "cuda_device_tokens",
    "default_cuda_device_groups",
    "discover_cuda_device_tokens",
    "normalize_cuda_device_groups",
]
