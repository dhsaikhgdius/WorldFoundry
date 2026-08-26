"""NK: kernels package init stays torch-free until heavy symbols are touched."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_kernels_package_import_does_not_load_torch() -> None:
    script = """
import sys
import worldfoundry.core.kernels as kernels
assert "torch" not in sys.modules
# native_provider submodule must also stay torch-free
status = kernels.native_provider_status(load=False, strict=False)
assert "torch" not in sys.modules
assert status.state == "absent"
# Touching a diffusion export may load torch; that is expected and out of scope.
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_kernels_lazy_export_still_resolves_diffusion_symbol() -> None:
    script = """
import worldfoundry.core.kernels as kernels
fn = kernels.residual_gate_add
assert callable(fn)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
