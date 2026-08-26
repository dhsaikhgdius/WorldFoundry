"""DX: Makefile exposes layered check-fast / check / check-ci gates."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def _makefile_text() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def test_makefile_declares_check_layer_targets() -> None:
    text = _makefile_text()
    for target in ("check-fast:", "check:", "check-ci:", "cli-check:", "lint:"):
        assert target in text, f"missing Makefile target {target}"


def test_makefile_check_layers_wire_expected_deps() -> None:
    text = _makefile_text()
    # Dependency lines may wrap; match the phony recipe headers only.
    assert "check-fast: ruff-check shell-check docs-check" in text
    assert "check: lint docs-check" in text
    assert "check-ci: check cli-check" in text


def test_makefile_help_mentions_dx_layers() -> None:
    text = _makefile_text()
    assert "make check-fast" in text
    assert "make check-ci" in text
    assert "CPU evaluate hello-world" in text
