"""C-05: Makefile syntax-check replaces the misnamed format-check gate."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def _target_body(text: str, target: str) -> str:
    """Return the recipe block that starts at a line-anchored target definition."""
    marker = f"\n{target}:"
    assert marker in text, f"Makefile target {target!r} is missing"
    after = text.split(marker, 1)[1]
    return after.split("\n\n", 1)[0]


def test_c05_lint_uses_syntax_check() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    lint_line = next(line for line in text.splitlines() if line.startswith("lint:"))
    deps = lint_line.split(":", 1)[1].split()
    assert "syntax-check" in deps
    # The misnamed gate must no longer be a direct lint dependency.
    assert "format-check" not in deps


def test_c05_syntax_check_scope_is_canonical_diffusion_only() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    body = _target_body(text, "syntax-check")
    assert "compileall" in body
    assert "$(CANONICAL_DIFFUSION_SOURCES)" in body
    # worldfoundry/evaluation and scripts are compile-eval's job; keep them out.
    assert "worldfoundry/evaluation" not in body
    body_recipe_lines = [line for line in body.splitlines() if line.startswith("\t")]
    assert not any("scripts" in line for line in body_recipe_lines)


def test_c05_format_check_is_compat_alias() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    alias_lines = [line for line in text.splitlines() if line.startswith("format-check:")]
    assert alias_lines == ["format-check: syntax-check"]


def test_c05_ruff_format_check_target_exists() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    body = _target_body(text, "ruff-format-check")
    assert "ruff format --check" in body
    assert "$(RUFF_SOURCES)" in body
