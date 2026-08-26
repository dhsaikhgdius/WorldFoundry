"""LG-01: Studio workspace main wires configure_logging (source contract)."""

from __future__ import annotations

from pathlib import Path


def test_workspace_main_source_calls_configure_logging() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "worldfoundry"
        / "studio"
        / "workspace_app.py"
    )
    text = path.read_text(encoding="utf-8")
    # Locate main() without importing FastAPI-heavy module.
    start = text.index("\ndef main(")
    end = text.index("\nif __name__", start)
    main_src = text[start:end]
    assert "configure_logging" in main_src
    assert "from worldfoundry.core.logging_setup import configure_logging" in main_src
