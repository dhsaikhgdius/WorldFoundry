"""SY-06: packaged ``*_runtime`` trees must carry UPSTREAM + license artifacts."""

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


def _packaged_runtime_dirs() -> list[Path]:
    payload = yaml.safe_load(RUNTIME_SOURCE.read_text(encoding="utf-8"))
    return [REPO_ROOT / item for item in payload["packaged"]]


def test_packaged_runtimes_have_upstream_and_license_artifacts():
    missing: list[str] = []
    for runtime_dir in _packaged_runtime_dirs():
        assert runtime_dir.is_dir(), f"missing packaged runtime: {runtime_dir}"
        upstream = runtime_dir / "UPSTREAM.md"
        if not upstream.is_file():
            missing.append(f"{runtime_dir.relative_to(REPO_ROOT)}: missing UPSTREAM.md")
        if not any((runtime_dir / name).is_file() for name in _LICENSE_NAMES):
            missing.append(
                f"{runtime_dir.relative_to(REPO_ROOT)}: missing license artifact "
                f"(one of {_LICENSE_NAMES})"
            )
    assert not missing, "SY-06 provenance gaps:\n" + "\n".join(missing)
