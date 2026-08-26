"""Tests for CUDA_DEVICE_ORDER pinning."""

from __future__ import annotations

from worldfoundry.runtime.device_pool import ensure_cuda_device_order


def test_ensure_cuda_device_order_setdefault() -> None:
    env: dict[str, str] = {}
    assert ensure_cuda_device_order(env) == "PCI_BUS_ID"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_ensure_cuda_device_order_preserves_explicit() -> None:
    env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    assert ensure_cuda_device_order(env) == "FASTEST_FIRST"
    assert env["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
