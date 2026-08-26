"""C-05: format-check renamed to syntax-check; ruff format --check available."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def test_c05_syntax_check_renames_format_check() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "syntax-check:" in text
    assert "format-check: syntax-check" in text
    assert "ruff-format-check:" in text
    assert "ruff format --check $(RUFF_SOURCES)" in text
    # Narrow compileall scope (no worldfoundry/evaluation scripts dump).
    assert (
        "compileall -q $(CANONICAL_DIFFUSION_SOURCES)" in text
        or "compileall -q $(CANONICAL_DIFFUSION_SOURCES)\n" in text
    )
    assert "compileall -q $(CANONICAL_DIFFUSION_SOURCES) worldfoundry/evaluation scripts" not in text
    # Preserve D-08 widened shell-check.
    assert "docs/fumadocs/scripts" in text or "scripts/dev" in text
    assert "lint: ruff-check syntax-check shell-check" in text
