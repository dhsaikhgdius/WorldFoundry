"""Tests for CUDA torch constraint stub dry-run checker."""

from __future__ import annotations

from pathlib import Path

from scripts.setup.check_cuda_torch_constraints import check_tier, main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_check_tier_ok_on_repo_stubs() -> None:
    path = REPO_ROOT / "requirements/cuda/cu128-torch.txt"
    assert check_tier(path) == []


def test_main_ok() -> None:
    assert main([]) == 0
