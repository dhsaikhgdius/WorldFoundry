from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "scripts" / "setup" / "default_world_checkpoint_links.yaml"


def test_default_world_checkpoint_links_manifest_shape():
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    repos = payload["repos"]
    assert isinstance(repos, list) and len(repos) >= 50
    seen: set[str] = set()
    for item in repos:
        repo_id = item["repo_id"]
        local_dir = item["local_dir"]
        assert repo_id and local_dir
        assert "=" not in repo_id and "=" not in local_dir
        assert repo_id not in seen
        seen.add(repo_id)
