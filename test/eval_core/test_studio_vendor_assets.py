"""PK-01: Studio vendor asset manifest + missing-file hint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.studio.ui.assets import (
    VENDOR_FETCH_HINT,
    VENDOR_MANIFEST_PATH,
    missing_vendor_module_paths,
    require_vendor_modules,
    vendor_manifest_assets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_vendor_manifest_is_pinned_and_complete() -> None:
    assert VENDOR_MANIFEST_PATH.is_file()
    payload = json.loads(VENDOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = payload["assets"]
    assert {asset["id"] for asset in assets} == {
        "spark.module.min.js",
        "three.module.js",
        "three.core.js",
    }
    for asset in assets:
        assert asset["url"].startswith("https://")
        assert len(asset["sha256"]) == 64
        assert Path(asset["relative_path"]).parts[0] in {"spark", "three"}


@pytest.mark.unit
def test_require_vendor_modules_hints_fetch_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("worldfoundry.studio.ui.assets.VENDOR_DIR", tmp_path / "vendor")
    missing = missing_vendor_module_paths()
    assert missing
    with pytest.raises(FileNotFoundError, match="fetch_vendor_assets"):
        require_vendor_modules()
    assert "scripts/studio/fetch_vendor_assets.py" in VENDOR_FETCH_HINT
    assert vendor_manifest_assets()
    assert (REPO_ROOT / "scripts" / "studio" / "fetch_vendor_assets.py").is_file()
