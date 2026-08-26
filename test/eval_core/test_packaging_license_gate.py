"""Release-gate tests for the packaging license gate (code_review VI-16).

Wheel content is decided by ``[tool.setuptools].packages.find`` alone;
MANIFEST.in ``prune`` lines only shape the sdist. These tests pin the two
invariants that previously rotted silently: no dead exclude patterns, and no
license-gated vendored tree in the wheel package set.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.setup.check_packaging_license_gate import (
    audit_wheel,
    dead_exclude_patterns,
    discover_packages,
    leaked_packages,
    license_gated_paths,
    license_gated_prefixes,
    load_find_config,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_dead_exclude_patterns() -> None:
    find_config = load_find_config()
    all_packages = discover_packages(find_config, apply_exclude=False)
    dead = dead_exclude_patterns(all_packages, list(find_config["exclude"]))
    assert dead == [], f"dead exclude patterns in pyproject.toml: {dead}"


def test_no_license_gated_packages_in_wheel_set() -> None:
    find_config = load_find_config()
    kept = discover_packages(find_config, apply_exclude=True)
    leaks = leaked_packages(kept, license_gated_prefixes())
    assert leaks == [], f"license-gated packages leak into the wheel set: {leaks}"


def test_manifest_license_gate_block_parses() -> None:
    prefixes = license_gated_prefixes()
    assert "worldfoundry.base_models.three_dimensions.general_3d.dust3r" in prefixes
    assert (
        "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay"
        in prefixes
    )
    assert len(prefixes) >= 15
    # thirdparty/ checkouts are not importable packages and must be skipped.
    assert all(prefix.startswith("worldfoundry.") for prefix in prefixes)


def test_gated_trees_still_present_as_local_checkouts() -> None:
    """The gate excludes trees from the wheel; it must not require deleting
    them from the working copy."""
    for path in ("worldfoundry/base_models/three_dimensions/general_3d/dust3r",):
        assert (REPO_ROOT / path).is_dir()


def test_audit_wheel_flags_and_passes(tmp_path: Path) -> None:
    gated = license_gated_paths()
    dirty = tmp_path / "dirty.whl"
    with zipfile.ZipFile(dirty, "w") as wheel:
        wheel.writestr(f"{gated[0]}/__init__.py", "")
        wheel.writestr("worldfoundry/__init__.py", "")
    assert audit_wheel(dirty, gated) == [f"{gated[0]}/__init__.py"]

    clean = tmp_path / "clean.whl"
    with zipfile.ZipFile(clean, "w") as wheel:
        wheel.writestr("worldfoundry/__init__.py", "")
    assert audit_wheel(clean, gated) == []


def test_main_ok() -> None:
    assert main([]) == 0
