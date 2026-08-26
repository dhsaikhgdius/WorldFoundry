"""Contract: dead conda_profile_install + ignored legacy install flags are gone."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP = REPO_ROOT / "scripts" / "setup"


def test_conda_profile_install_removed() -> None:
    assert not (SETUP / "conda_profile_install.sh").exists()


def test_legacy_ignored_flags_removed_from_installers() -> None:
    for name in ("conda_install.sh", "unified_install.sh"):
        text = (SETUP / name).read_text(encoding="utf-8")
        assert "--pytorch-bundle" not in text
        assert "--transformers" not in text
        assert "--three-d-core" not in text
        assert "--skip-three-d-core" not in text
