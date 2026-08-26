"""C-08 leftovers: cancel-in-progress false; honest Pages prep step; skip-bootstrap no-op."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = REPO_ROOT / ".github" / "workflows" / "deploy-docs.yml"
BUILD_SH = REPO_ROOT / "scripts" / "docs" / "build.sh"


def test_c08_deploy_docs_does_not_cancel_in_progress() -> None:
    payload = yaml.safe_load(DEPLOY.read_text(encoding="utf-8"))
    assert payload["concurrency"]["cancel-in-progress"] is False


def test_c08_pages_prep_step_name_matches_behavior() -> None:
    text = DEPLOY.read_text(encoding="utf-8")
    assert "Drop demos from Pages artifact and add .nojekyll" in text
    assert "Preserve Next.js assets" not in text
    assert "rm -rf docs/fumadocs/out/demos" in text
    assert "touch docs/fumadocs/out/.nojekyll" in text


def test_c08_skip_bootstrap_is_accepted_noop() -> None:
    text = BUILD_SH.read_text(encoding="utf-8")
    assert "SKIP_BOOTSTRAP" not in text
    assert "--skip-bootstrap)" in text
    assert "historical no-op" in text.lower() or "Historical no-op" in text
    # Empty bootstrap block removed.
    assert 'if [[ "${SKIP_BOOTSTRAP}"' not in text
