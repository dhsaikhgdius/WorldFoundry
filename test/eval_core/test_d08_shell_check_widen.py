"""D-08: Makefile shell-check covers docker/embodied/test (+ fumadocs/dev) scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def test_d08_shell_check_covers_docker_and_embodied() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    # Locate the shell-check recipe body.
    after = text.split("shell-check:", 1)[1]
    body = after.split("\n\n", 1)[0]
    assert "docker" in body
    assert "scripts/embodied" in body
    assert "test" in body
    assert "bash -n" in body
    # Must not still be setup-only.
    assert "find scripts/setup -type f -name '*.sh' -exec bash -n {} +" not in body
