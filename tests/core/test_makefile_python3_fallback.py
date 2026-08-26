"""DX: Makefile falls back to python3 when python is absent."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_makefile_falls_back_to_python3_when_python_missing() -> None:
    make_bin = shutil.which("make")
    assert make_bin is not None
    with tempfile.TemporaryDirectory() as tmp:
        bin_dir = Path(tmp) / "bin"
        bin_dir.mkdir()
        python3 = shutil.which("python3")
        assert python3 is not None
        (bin_dir / "python3").symlink_to(python3)
        # Deliberately omit `python` from PATH, but keep `make` resolvable.
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{Path(make_bin).parent}"
        completed = subprocess.run(
            [make_bin, "-C", str(REPO_ROOT), "-n", "docs-check"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "python3 -m worldfoundry.cli" in completed.stdout
        # Ensure we did not accidentally keep a bare `python` interpreter token.
        for line in completed.stdout.splitlines():
            if "worldfoundry.cli" in line:
                assert line.lstrip().startswith("PYTHONPATH=") or "python3 -m worldfoundry.cli" in line
                assert "python -m worldfoundry.cli" not in line.replace("python3 -m worldfoundry.cli", "")
