from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from worldfoundry.runtime import device_pool
from worldfoundry.runtime.device_pool import CudaDeviceLeasePool


@pytest.fixture(autouse=True)
def _disable_gpu_memory_warn_by_default(monkeypatch):
    # Keep lease unit tests hermetic unless a case opts into nvidia-smi.
    monkeypatch.setenv("WORLDFOUNDRY_GPU_MEMORY_WARN", "0")


def test_container_local_indices_override_host_nvidia_visible_devices(monkeypatch):
    monkeypatch.setattr(device_pool.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        device_pool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0\n1\n2\n"),
    )

    assert device_pool.discover_cuda_device_tokens(
        {"NVIDIA_VISIBLE_DEVICES": "7,0,1"}
    ) == ("0", "1", "2")


def test_explicit_cuda_visibility_remains_authoritative(monkeypatch):
    monkeypatch.setattr(
        device_pool.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("nvidia-smi should not run")),
    )

    assert device_pool.discover_cuda_device_tokens(
        {"CUDA_VISIBLE_DEVICES": "4,2", "NVIDIA_VISIBLE_DEVICES": "7,0"}
    ) == ("4", "2")


def test_cuda_device_leases_are_non_overlapping_and_reusable():
    pool = CudaDeviceLeasePool(("0", "1"))

    first = pool.acquire()
    second = pool.acquire()
    assert first.tokens == ("0",)
    assert second.tokens == ("1",)
    assert pool.available_count == 0

    first.release()
    replacement = pool.acquire()
    assert replacement.tokens == ("0",)

    second.release()
    replacement.release()
    assert pool.available_count == 2


def test_cuda_device_lease_waiter_wakes_after_release():
    pool = CudaDeviceLeasePool(("0",))
    first = pool.acquire()
    acquired: list[str] = []
    ready = threading.Event()

    def wait_for_device() -> None:
        ready.set()
        with pool.acquire() as lease:
            acquired.append(lease.visible_devices)

    thread = threading.Thread(target=wait_for_device)
    thread.start()
    assert ready.wait(timeout=1)
    deadline = time.monotonic() + 1
    while pool.waiting_count == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.waiting_count == 1
    first.release()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert acquired == ["0"]


def test_warn_if_gpu_memory_high_logs_when_ratio_exceeded(monkeypatch, caplog):
    monkeypatch.setenv("WORLDFOUNDRY_GPU_MEMORY_WARN", "1")
    monkeypatch.setenv("WORLDFOUNDRY_GPU_MEMORY_WARN_RATIO", "0.5")
    monkeypatch.setattr(device_pool.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        device_pool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="0, GPU-aaa, 9000, 10000\n1, GPU-bbb, 100, 10000\n",
        ),
    )
    with caplog.at_level(logging.WARNING, logger="worldfoundry.runtime.device_pool"):
        hits = device_pool.warn_if_gpu_memory_high(("0", "1"))
    assert len(hits) == 1
    assert hits[0]["token"] == "0"
    assert "memory high" in caplog.text
