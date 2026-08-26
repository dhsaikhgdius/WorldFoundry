"""Import-light CUDA device discovery and worker grouping helpers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable

_DISABLED_DEVICE_VALUES = frozenset({"", "-1", "none", "void"})
_ALL_DEVICE_VALUES = frozenset({"all"})
_FALSEY = frozenset({"0", "false", "no", "off"})
_logger = logging.getLogger(__name__)


def _gpu_memory_warn_enabled() -> bool:
    raw = str(os.environ.get("WORLDFOUNDRY_GPU_MEMORY_WARN", "1")).strip().lower()
    return raw not in _FALSEY


def _gpu_memory_warn_ratio() -> float:
    raw = str(os.environ.get("WORLDFOUNDRY_GPU_MEMORY_WARN_RATIO", "0.9")).strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.9
    return min(max(value, 0.0), 1.0)


def warn_if_gpu_memory_high(
    tokens: Sequence[str],
    *,
    timeout_seconds: float = 2.0,
) -> list[dict[str, float | str]]:
    """Warn when leased GPU tokens already exceed a used-memory ratio.

    Uses ``nvidia-smi`` (no Torch / NVML import). Disabled with
    ``WORLDFOUNDRY_GPU_MEMORY_WARN=0``. Ratio threshold defaults to ``0.9``
    (``WORLDFOUNDRY_GPU_MEMORY_WARN_RATIO``).
    """

    if not tokens or not _gpu_memory_warn_enabled():
        return []
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_seconds), 0.1),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if completed.returncode != 0:
        return []

    wanted = {str(token).strip() for token in tokens if str(token).strip()}
    threshold = _gpu_memory_warn_ratio()
    hits: list[dict[str, float | str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index, uuid, used_raw, total_raw = parts[0], parts[1], parts[2], parts[3]
        if index not in wanted and uuid not in wanted:
            continue
        try:
            used = float(used_raw)
            total = float(total_raw)
        except ValueError:
            continue
        if total <= 0:
            continue
        ratio = used / total
        if ratio < threshold:
            continue
        payload = {
            "token": index if index in wanted else uuid,
            "used_mib": used,
            "total_mib": total,
            "ratio": ratio,
        }
        hits.append(payload)
        _logger.warning(
            "CUDA device %s memory high before lease use: %.0f/%.0f MiB (%.0f%% >= %.0f%%)",
            payload["token"],
            used,
            total,
            ratio * 100.0,
            threshold * 100.0,
        )
    return hits


@dataclass
class CudaDeviceLease:
    """A reversible reservation of one or more physical CUDA device tokens."""

    _pool: "CudaDeviceLeasePool" = field(repr=False)
    tokens: tuple[str, ...]
    _released: bool = field(default=False, init=False, repr=False)

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
        self._pool.release(self.tokens)

    def __enter__(self) -> "CudaDeviceLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class CudaDeviceLeasePool:
    """Thread-safe allocator for non-overlapping CUDA worker assignments."""

    def __init__(self, devices: Sequence[str]) -> None:
        normalized = normalize_cuda_device_groups(tuple(str(device) for device in devices))
        self._devices = tuple(normalized)
        self._available = list(self._devices)
        self._leased: set[str] = set()
        self._waiting = 0
        self._condition = threading.Condition()

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
        with self._condition:
            self._waiting += 1
            try:
                while len(self._available) < requested:
                    if cancel_requested is not None and cancel_requested():
                        raise RuntimeError("CUDA device allocation cancelled")
                    self._condition.wait(timeout=max(float(poll_interval), 0.01))
                tokens = tuple(self._available[:requested])
                del self._available[:requested]
                self._leased.update(tokens)
            finally:
                self._waiting -= 1
        warn_if_gpu_memory_high(tokens)
        return CudaDeviceLease(self, tokens)

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
    "warn_if_gpu_memory_high",
]
