"""SY-06: ``*_runtime`` trees must carry UPSTREAM + license artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SOURCE = REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "runtime_source_manifest.yaml"

_LICENSE_NAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENSE.NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_LICENSES.md",
)


def _manifest() -> dict:
    return yaml.safe_load(RUNTIME_SOURCE.read_text(encoding="utf-8"))


def _packaged_runtime_dirs() -> list[Path]:
    return [REPO_ROOT / item for item in _manifest()["packaged"]]


def _checkout_only_provenance_dirs() -> list[Path]:
    return [REPO_ROOT / item for item in _manifest().get("checkout_only_provenance") or ()]


def _assert_provenance(runtime_dirs: list[Path], *, label: str) -> None:
    missing: list[str] = []
    for runtime_dir in runtime_dirs:
        assert runtime_dir.is_dir(), f"missing {label} runtime: {runtime_dir}"
        upstream = runtime_dir / "UPSTREAM.md"
        if not upstream.is_file():
            missing.append(f"{runtime_dir.relative_to(REPO_ROOT)}: missing UPSTREAM.md")
        if not any((runtime_dir / name).is_file() for name in _LICENSE_NAMES):
            missing.append(
                f"{runtime_dir.relative_to(REPO_ROOT)}: missing license artifact "
                f"(one of {_LICENSE_NAMES})"
            )
    assert not missing, f"SY-06 {label} provenance gaps:\n" + "\n".join(missing)


def test_packaged_runtimes_have_upstream_and_license_artifacts():
    _assert_provenance(_packaged_runtime_dirs(), label="packaged")


def test_checkout_only_provenance_cohort_has_upstream_and_license_artifacts():
    dirs = _checkout_only_provenance_dirs()
    assert dirs, "checkout_only_provenance cohort must be non-empty"
    packaged = {path.resolve() for path in _packaged_runtime_dirs()}
    overlap = [path for path in dirs if path.resolve() in packaged]
    assert not overlap, f"checkout_only_provenance overlaps packaged: {overlap}"
    _assert_provenance(dirs, label="checkout_only")
