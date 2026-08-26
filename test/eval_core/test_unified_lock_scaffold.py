"""Contract tests for the per-tier unified lock scaffolding (plan I-05)."""

from __future__ import annotations

from pathlib import Path

from scripts.setup.check_unified_lock import (
    check_clip_pin,
    check_lock_header,
    check_lock_pins,
    check_tier_lock,
    lock_is_populated,
    main,
)
from worldfoundry.runtime.cuda_tiers import SUPPORTED_CUDA_TIERS

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lock_scaffold_exists_for_all_tiers() -> None:
    for tier in SUPPORTED_CUDA_TIERS:
        path = REPO_ROOT / f"requirements/lock/worldfoundry-unified.{tier}.lock.txt"
        assert path.is_file(), f"missing lock scaffold for {tier}"
        errors, _populated = check_tier_lock(REPO_ROOT, tier)
        assert errors == []


def test_main_ok_on_repo() -> None:
    assert main([]) == 0


def test_clip_git_dependency_is_sha_pinned() -> None:
    assert check_clip_pin(REPO_ROOT) == []


def test_header_rejects_wrong_tier_index() -> None:
    text = "# --index-url https://download.pytorch.org/whl/cu124\n"
    path = Path("worldfoundry-unified.cu121.lock.txt")
    errors = check_lock_header(text, tier="cu121", path=path)
    assert any("cu121" in err for err in errors)
    assert any("cu124" in err for err in errors)


def test_populated_lock_requires_torch_pins_inside_tier_bounds() -> None:
    path = Path("worldfoundry-unified.cu121.lock.txt")
    # cu121 caps torch at <2.6 (I-03); a 2.7 pin means the resolve fell
    # through to an index that does not serve this tier.
    text = (
        "# --index-url https://download.pytorch.org/whl/cu121\n"
        "torch==2.7.0\n"
        "torchvision==0.20.0+cu121\n"
        "torchaudio==2.5.1+cu121\n"
    )
    assert lock_is_populated(text)
    errors = check_lock_pins(text, tier="cu121", path=path)
    assert any("torch==2.7.0" in err for err in errors)

    ok_text = (
        "# --index-url https://download.pytorch.org/whl/cu121\n"
        "torch==2.5.1+cu121\n"
        "torchvision==0.20.1+cu121\n"
        "torchaudio==2.5.1+cu121\n"
    )
    assert check_lock_pins(ok_text, tier="cu121", path=path) == []


def test_comment_only_lock_counts_as_placeholder() -> None:
    placeholder = "# WorldFoundry unified lock for CUDA tier cu128 (plan I-05).\n#\n\n"
    assert not lock_is_populated(placeholder)
    assert lock_is_populated(placeholder + "torch==2.7.0+cu128\n")
