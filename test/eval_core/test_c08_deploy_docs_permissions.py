"""C-08: deploy-docs scopes pages/id-token write to the deploy job."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-docs.yml"


def test_c08_deploy_docs_scopes_id_token_to_deploy_job() -> None:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    top = payload.get("permissions") or {}
    assert top.get("contents") == "read"
    assert "id-token" not in top
    assert top.get("pages") != "write"

    build = payload["jobs"]["build"].get("permissions") or {}
    assert build.get("pages") == "write"
    assert "id-token" not in build

    deploy = payload["jobs"]["deploy"].get("permissions") or {}
    assert deploy.get("pages") == "write"
    assert deploy.get("id-token") == "write"
