"""DX: shell-check covers setup, scripts/dev, and fumadocs shell scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def test_shell_check_covers_dev_and_fumadocs_scripts() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    start = text.index("shell-check:")
    end = text.index("\ndata-check:", start)
    block = text[start:end]
    assert "scripts/setup" in block
    assert "scripts/dev" in block
    assert "docs/fumadocs/scripts" in block
    assert "bash -n" in block
