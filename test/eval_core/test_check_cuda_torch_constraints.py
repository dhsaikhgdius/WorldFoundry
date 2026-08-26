"""Tests for CUDA torch constraint stub dry-run checker (plan I-03)."""

from __future__ import annotations

from pathlib import Path

from scripts.setup.check_cuda_torch_constraints import check_tier, main
from worldfoundry.runtime.cuda_tiers import TIER_TORCH_SPECS, torch_specs_for_tier

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_check_tier_ok_on_repo_stubs() -> None:
    for tier, expected in TIER_TORCH_SPECS.items():
        path = REPO_ROOT / f"requirements/cuda/{tier}-torch.txt"
        assert check_tier(path, expected=expected) == []


def test_main_ok() -> None:
    assert main([]) == 0


def test_torch_specs_for_tier_matches_ssot() -> None:
    assert torch_specs_for_tier("cu121")["torch"] == "torch>=2.4,<2.6"
    assert torch_specs_for_tier("cu124")["torch"] == "torch>=2.4,<2.7"
    assert torch_specs_for_tier("cu128")["torch"] == "torch>=2.7,<2.12.0"
