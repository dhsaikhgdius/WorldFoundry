"""DO: fumadocs catalog-coverage has generate/check npm scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FUMADOCS = REPO_ROOT / "docs" / "fumadocs"
SCRIPT = FUMADOCS / "scripts" / "generate-catalog-coverage.py"


def test_package_json_exposes_catalog_scripts() -> None:
    scripts = json.loads((FUMADOCS / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert "catalog:generate" in scripts
    assert "catalog:check" in scripts
    assert "generate-catalog-coverage.py" in scripts["catalog:generate"]
    assert "--check" in scripts["catalog:check"]


def test_catalog_coverage_check_mode_passes_on_current_tree() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=FUMADOCS,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
