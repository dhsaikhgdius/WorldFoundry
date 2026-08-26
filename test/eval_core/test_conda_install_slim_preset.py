"""Smoke checks that --preset slim is a real install path, not an alias."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONDA_INSTALL = REPO_ROOT / "scripts/setup/conda_install.sh"
UNIFIED_INSTALL = REPO_ROOT / "scripts/setup/unified_install.sh"


def test_slim_preset_installs_editable_extras_not_unified_only() -> None:
    text = CONDA_INSTALL.read_text(encoding="utf-8")
    assert 'INSTALL_PRESET" == "slim"' in text
    assert ".[tui,ui,api,hf]" in text
    assert "worldfoundry-unified.txt" in text
    assert "FLASH_ATTN_CLI_SET" in text
    # I-02 torch constraint must still wrap both preset install paths.
    assert "--constraint" in text
    assert 'TORCH_CONSTRAINT_FILE' in text


def test_unified_install_documents_slim_extras() -> None:
    text = UNIFIED_INSTALL.read_text(encoding="utf-8")
    assert "slim installs editable extras" in text
