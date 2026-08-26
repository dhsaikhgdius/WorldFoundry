"""DO: fumadocs codegen uses a portable Python launcher."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FUMADOCS = REPO_ROOT / "docs" / "fumadocs"
LAUNCHER = FUMADOCS / "scripts" / "run-python.sh"


def test_package_json_uses_run_python_launcher() -> None:
    payload = json.loads((FUMADOCS / "package.json").read_text(encoding="utf-8"))
    scripts = payload["scripts"]
    for key in ("api:check", "api:generate", "cli:screenshots", "models:generate"):
        assert "run-python.sh" in scripts[key]
        assert scripts[key].startswith("bash scripts/run-python.sh")


def test_run_python_honors_wf_docs_python_override(tmp_path: Path) -> None:
    assert LAUNCHER.is_file()
    marker = tmp_path / "marker.py"
    marker.write_text("print('ok-from-override')\n", encoding="utf-8")
    # Point WF_DOCS_PYTHON at the real interpreter but verify the env var is used
    # by wrapping through a tiny shim that prints a distinctive token.
    shim = tmp_path / "shim-python"
    real_py = shutil.which("python3") or shutil.which("python")
    assert real_py is not None
    shim.write_text(f"#!/usr/bin/env bash\nexec '{real_py}' \"$@\"\n", encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), str(marker)],
        cwd=FUMADOCS,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env={**os.environ, "WF_DOCS_PYTHON": str(shim), "PYTHON": "missing-python-should-not-be-used"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok-from-override" in completed.stdout
