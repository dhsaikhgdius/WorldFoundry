"""DO: docs-check covers CLI help plus zoo models and benchmarks."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def test_docs_check_includes_models_and_benchmarks() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "docs-check:" in text
    # Recipe must exercise both catalog surfaces (models was previously missing).
    start = text.index("docs-check:")
    end = text.index("\nlint:", start)
    block = text[start:end]
    assert "zoo models --json" in block
    assert "zoo benchmarks --json" in block
    assert "--help" in block


def test_docs_check_help_mentions_models() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "zoo models/benchmarks" in text or "models/benchmarks JSON" in text
