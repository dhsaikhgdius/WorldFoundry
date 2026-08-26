"""Actions workflows pin third-party actions to full commit SHAs."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SHA_RE = re.compile(r"uses:\s+[^\s]+@[0-9a-f]{40}\b")
TAG_RE = re.compile(r"uses:\s+actions/[^\s]+@v\d")


def test_ci_and_deploy_docs_pin_action_shas() -> None:
    for name in ("ci.yml", "deploy-docs.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert SHA_RE.search(text), name
        assert not TAG_RE.search(text), f"{name} still uses floating @v tags"
