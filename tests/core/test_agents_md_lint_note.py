"""C-01 / DX: AGENTS.md no longer claims the stale ~8 ruff F401 failures."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS = REPO_ROOT / "AGENTS.md"


def test_agents_md_does_not_claim_stale_ruff_failures() -> None:
    text = AGENTS.read_text(encoding="utf-8")
    assert "~8 pre-existing failures" not in text
    assert "F401" not in text
    assert "make lint is green" in text or "ruff==0.12.7" in text
