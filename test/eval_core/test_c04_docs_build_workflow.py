"""C-04: path-filtered docs-build workflow exists and pins Action SHAs."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docs-build.yml"
TAG_RE = re.compile(r"uses:\s+actions/[^\s]+@v\d")


def test_c04_docs_build_workflow_path_filter() -> None:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert payload["name"] == "docs-build"
    # PyYAML parses the workflow key `on:` as boolean True.
    on_block = payload[True]
    paths = on_block["pull_request"]["paths"]
    assert "docs/**" in paths
    assert "scripts/docs/**" in paths
    assert ".github/workflows/docs-build.yml" in paths
    assert "build" in payload["jobs"]


def test_c04_docs_build_pins_action_shas() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert not TAG_RE.search(text)
    assert "bash scripts/docs/build.sh --skip-bootstrap" in text
