"""D-10: interactive toolbox lives in the ``dev`` stage, not ``base-devel``."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
BUILD = REPO_ROOT / "docker" / "build_with_docker.sh"

# Packages that must not bloat the default compile image (plan D-10).
_INTERACTIVE = ("nano", "tmux", "screen", "iputils-ping")


def _stage_body(text: str, stage: str) -> str:
    marker = f"AS {stage}"
    start = text.index(marker)
    rest = text[start:]
    # Next stage header or default export.
    next_headers = [
        rest.find("\nFROM ", 1),
        rest.find("\n# Default export"),
    ]
    ends = [i for i in next_headers if i > 0]
    end = min(ends) if ends else len(rest)
    return rest[:end]


def test_d10_interactive_tools_only_in_dev_stage() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "AS base-devel" in text
    assert "AS dev" in text

    devel = _stage_body(text, "base-devel")
    dev = _stage_body(text, "dev")

    for pkg in _INTERACTIVE:
        assert pkg not in devel, f"{pkg} must not be in base-devel (D-10)"
        assert pkg in dev, f"{pkg} must be installed in the dev stage"

    # Default export remains the compile image, not the interactive toolbox.
    assert "FROM base-devel" in text.split("# Default export")[-1]


def test_d10_build_script_documents_dev_target() -> None:
    text = BUILD.read_text(encoding="utf-8")
    assert "base-runtime | base-devel | dev | cpu" in text
    assert 'TARGET="${WORLDFOUNDRY_DOCKER_TARGET:-base-devel}"' in text
