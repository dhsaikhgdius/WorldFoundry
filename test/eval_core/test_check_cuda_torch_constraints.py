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


def _parse_ge_lt(spec: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse a simple ``pkg>=X,<Y`` / ``>=X,<Y`` pin into numeric bounds."""

    body = spec.split(";", 1)[0].strip()
    if body[0].isalpha():
        # strip package name
        for i, ch in enumerate(body):
            if ch in "<>=":
                body = body[i:]
                break
    lower = upper = None
    for part in body.split(","):
        part = part.strip()
        if part.startswith(">="):
            lower = tuple(int(p) for p in part[2:].split("."))
        elif part.startswith("<"):
            upper = tuple(int(p) for p in part[1:].split("."))
    if lower is None or upper is None:
        raise AssertionError(f"expected >=,< bounds in {spec!r}")
    return lower, upper


def test_optimized_core_torch_intersects_all_cuda_tiers() -> None:
    """I-03: unified extras must not force torch>=2.7 against cu121/cu124 floors."""

    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    # unified.txt pulls optimized_core; `all` embeds the same runtime torch floor.
    for extra in ("optimized_core", "all"):
        torch_pins = [
            req
            for req in extras[extra]
            if req == "torch"
            or req.startswith("torch>")
            or req.startswith("torch<")
            or req.startswith("torch=")
        ]
        assert len(torch_pins) == 1, (extra, torch_pins)
        extra_lo, extra_hi = _parse_ge_lt(torch_pins[0])
        for tier, specs in TIER_TORCH_SPECS.items():
            tier_lo, tier_hi = _parse_ge_lt(specs["torch"])
            # Ranges [lo, hi) intersect when lo < other_hi and other_lo < hi.
            assert extra_lo < tier_hi and tier_lo < extra_hi, (
                extra,
                tier,
                torch_pins[0],
                specs["torch"],
            )
