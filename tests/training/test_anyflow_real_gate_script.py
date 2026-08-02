from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _gate_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "training" / "validate_anyflow_roundtrip.py"
    spec = importlib.util.spec_from_file_location("validate_anyflow_roundtrip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_anyflow_gate_binds_audited_checkpoint_without_loading_weights() -> None:
    gate = _gate_module()

    identity = gate._checkpoint_identity()

    assert identity == (
        "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
        "@4c2ec05c7fa4dbafbca131ad32430905c7ff2974"
    )
    assert gate._far_checkpoint_identity() == (
        "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers"
        "@915af337434035df8545797ecc910d79fa78cf29"
    )
    assert gate._torch_dtype("bfloat16") is torch.bfloat16


def test_real_anyflow_gate_state_digest_is_byte_sensitive() -> None:
    gate = _gate_module()
    first = {
        "tensor": torch.tensor([1.0, 2.0]),
        "optimizer_step": torch.tensor(1.0),
    }
    second = {
        "tensor": torch.tensor([1.0, 3.0]),
        "optimizer_step": torch.tensor(1.0),
    }

    assert gate._state_digest(first) == gate._state_digest(first)
    assert gate._state_digest(first) != gate._state_digest(second)
