"""SY-05: ``*_runtime`` shipping contract (packaged vs checkout_only)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "runtime_source_manifest.yaml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
MANIFEST_IN = REPO_ROOT / "MANIFEST.in"


def _discover_runtime_dirs() -> list[Path]:
    """Return top-level ``*_runtime`` dirs (skip nested runtimes inside another)."""

    found: list[Path] = []
    for path in sorted((REPO_ROOT / "worldfoundry").rglob("*_runtime")):
        if not path.is_dir():
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part.endswith("_runtime") for part in rel.parts[:-1]):
            continue
        found.append(rel)
    return found


def _load_manifest() -> dict:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    packaged = [Path(item) for item in payload.get("packaged") or []]
    assert packaged, "runtime_source_manifest.packaged must be non-empty"
    return {"packaged": packaged, "raw": payload}


def _package_data_text() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("[tool.setuptools.package-data]")
    end = text.find("\n[tool.setuptools.exclude-package-data]", start)
    return text[start:end]


def _manifest_in_text() -> str:
    return MANIFEST_IN.read_text(encoding="utf-8")


def test_runtime_source_manifest_lists_only_existing_packaged_trees():
    manifest = _load_manifest()
    for path in manifest["packaged"]:
        assert (REPO_ROOT / path).is_dir(), f"packaged runtime missing on disk: {path}"


def test_every_discovered_runtime_is_classified():
    manifest = _load_manifest()
    packaged = {path.as_posix() for path in manifest["packaged"]}
    discovered = {path.as_posix() for path in _discover_runtime_dirs()}
    unknown_packaged = packaged - discovered
    assert not unknown_packaged, f"packaged entries not discovered as *_runtime: {sorted(unknown_packaged)}"
    # checkout_only is the default for the remainder — no orphan requirement beyond coverage.
    assert packaged <= discovered


def test_packaged_runtimes_appear_in_package_data_and_manifest_in():
    manifest = _load_manifest()
    package_data = _package_data_text()
    manifest_in = _manifest_in_text()
    missing_pd: list[str] = []
    missing_mi: list[str] = []
    for path in manifest["packaged"]:
        name = path.name
        if name not in package_data and path.as_posix() not in package_data:
            missing_pd.append(path.as_posix())
        # MANIFEST.in uses recursive-include <path> ...
        if path.as_posix() not in manifest_in:
            missing_mi.append(path.as_posix())
    assert not missing_pd, f"packaged runtimes missing from package-data: {missing_pd}"
    assert not missing_mi, f"packaged runtimes missing from MANIFEST.in: {missing_mi}"


def test_checkout_only_runtimes_are_not_accidentally_in_package_data():
    """Spot-check: a known checkout_only tree must not be listed in package-data."""

    package_data = _package_data_text()
    # open_sora_runtime is pruned in MANIFEST and excluded from find_packages patterns.
    assert "open_sora_runtime" not in package_data
    assert "wonderworld_runtime" not in package_data
    assert "ac3d_runtime" not in package_data


def test_manifest_schema_version_present():
    payload = _load_manifest()["raw"]
    assert int(payload.get("schema_version") or 0) >= 1
