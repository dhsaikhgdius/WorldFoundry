from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from worldfoundry.runtime import device_pool
from worldfoundry.runtime.device_pool import CudaDeviceLeasePool


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


def test_cuda_device_leases_are_non_overlapping_and_reusable(tmp_path: Path):
    pool = CudaDeviceLeasePool(("0", "1"), locks_dir=tmp_path)

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


def test_cuda_device_lease_waiter_wakes_after_release(tmp_path: Path):
    pool = CudaDeviceLeasePool(("0",), locks_dir=tmp_path)
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


def test_fifo_ticket_prevents_small_request_starvation(tmp_path: Path):
    pool = CudaDeviceLeasePool(("0", "1"), locks_dir=tmp_path)
    held = [pool.acquire(), pool.acquire()]
    order: list[str] = []
    large_ready = threading.Event()
    small_ready = threading.Event()

    def large() -> None:
        large_ready.set()
        with pool.acquire(2) as lease:
            order.append(f"large:{lease.visible_devices}")

    def small() -> None:
        # Ensure large is already waiting at the head of the ticket queue.
        assert large_ready.wait(timeout=1)
        deadline = time.monotonic() + 1
        while pool.waiting_count < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        small_ready.set()
        with pool.acquire(1) as lease:
            order.append(f"small:{lease.visible_devices}")

    t_large = threading.Thread(target=large)
    t_small = threading.Thread(target=small)
    t_large.start()
    t_small.start()
    assert small_ready.wait(timeout=1)
    deadline = time.monotonic() + 1
    while pool.waiting_count < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    for lease in held:
        lease.release()
    t_large.join(timeout=2)
    t_small.join(timeout=2)
    assert not t_large.is_alive() and not t_small.is_alive()
    assert order[0].startswith("large:")
    assert order[1].startswith("small:")


def _cross_process_worker(locks_dir: str, hold_seconds: float, result_queue: mp.Queue) -> None:
    pool = CudaDeviceLeasePool(("gpu-x",), locks_dir=Path(locks_dir), cross_process=True)
    started = time.monotonic()
    with pool.acquire() as lease:
        result_queue.put(("acquired", lease.tokens[0], time.monotonic() - started))
        time.sleep(hold_seconds)
    result_queue.put(("released", lease.tokens[0], time.monotonic() - started))


def test_cross_process_flock_is_exclusive(tmp_path: Path):
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    first = ctx.Process(target=_cross_process_worker, args=(str(tmp_path), 0.4, queue))
    second = ctx.Process(target=_cross_process_worker, args=(str(tmp_path), 0.05, queue))
    first.start()
    time.sleep(0.05)
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    assert first.exitcode == 0 and second.exitcode == 0
    events = [queue.get(timeout=1) for _ in range(4)]
    acquired_waits = sorted(wait for kind, _token, wait in events if kind == "acquired")
    # Second process must wait until the first releases the flock.
    assert acquired_waits[1] >= 0.3
