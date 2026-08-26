"""DO: fumadocs models:check gates stale model-recipes JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FUMADOCS = REPO_ROOT / "docs" / "fumadocs"
SCRIPT = FUMADOCS / "scripts" / "generate-model-recipes.py"


def test_package_json_exposes_models_check() -> None:
    scripts = json.loads((FUMADOCS / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert "models:generate" in scripts
    assert "models:check" in scripts
    assert "--check" in scripts["models:check"]


def test_models_check_passes_on_current_tree() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=FUMADOCS,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
