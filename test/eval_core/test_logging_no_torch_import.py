"""CM-01: ``configure_logging`` must not drag ``torch`` into CLI processes.

``configure_logging`` used to import ``worldfoundry.core.distributed.logging``
to reparent the rank-aware singleton. That module imports ``torch`` at top
level and, until CM-01, its package ``__init__`` eagerly re-exported the whole
distributed stack — so every command that configured logging (explicit
``--log-level``, eager-logging commands, the MCP server) paid a multi-second
torch import. The reparenting is now driven by a ``sys.modules`` probe plus an
import-time hook on the distributed side, and the package re-exports are
PEP 562 lazy.

Each scenario runs in a subprocess so the parent test session's own imports
cannot contaminate ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_clean(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"


def test_configure_logging_does_not_import_torch_or_distributed() -> None:
    result = _run_python(
        """
import sys
from worldfoundry.core import configure_logging
configure_logging()
leaked = [
    name
    for name in sys.modules
    if name == "torch" or name.startswith("worldfoundry.core.distributed")
]
print("LEAKED=" + repr(sorted(leaked)))
"""
    )
    _assert_clean(result)
    assert "LEAKED=[]" in result.stdout, (
        "configure_logging must not import torch or the distributed stack; reparenting is a sys.modules probe (CM-01)"
    )


def test_distributed_package_import_stays_torch_free() -> None:
    result = _run_python(
        """
import sys
import worldfoundry.core.distributed
print("TORCH_LOADED=" + repr("torch" in sys.modules))
"""
    )
    _assert_clean(result)
    assert "TORCH_LOADED=False" in result.stdout, (
        "worldfoundry.core.distributed must re-export lazily (PEP 562); "
        "eager re-exports pull torch on package import (CM-01)"
    )


def test_cli_eager_logging_flag_does_not_import_torch() -> None:
    """Acceptance: ``--log-level`` forces eager configure_logging in the CLI."""
    result = _run_python(
        """
import sys
from worldfoundry.cli import main
exit_code = main(["--log-level", "INFO"])
print("TORCH_LOADED=" + repr("torch" in sys.modules))
raise SystemExit(exit_code)
"""
    )
    _assert_clean(result)
    assert "TORCH_LOADED=False" in result.stdout, (
        "a CLI invocation that configures logging must stay torch-free (CM-01)"
    )


def test_reparents_when_distributed_logging_loaded_first() -> None:
    pytest.importorskip("torch")
    result = _run_python(
        """
from worldfoundry.core.distributed.logging import distributed_logger
from worldfoundry.core import configure_logging
configure_logging()
print("PROPAGATE=" + repr(distributed_logger.propagate))
print("HANDLERS=" + repr(list(distributed_logger.handlers)))
"""
    )
    _assert_clean(result)
    assert "PROPAGATE=True" in result.stdout
    assert "HANDLERS=[]" in result.stdout


def test_adopts_pipeline_when_configured_before_import() -> None:
    pytest.importorskip("torch")
    result = _run_python(
        """
import logging
from worldfoundry.core import configure_logging
configure_logging(level="DEBUG")
from worldfoundry.core.distributed.logging import distributed_logger
print("PROPAGATE=" + repr(distributed_logger.propagate))
print("HANDLERS=" + repr(list(distributed_logger.handlers)))
print("LEVEL_MATCHES=" + repr(distributed_logger.level == logging.getLogger().level))
"""
    )
    _assert_clean(result)
    assert "PROPAGATE=True" in result.stdout, (
        "importing distributed.logging after configure_logging must reparent "
        "the singleton onto the unified pipeline (CM-01 companion hook)"
    )
    assert "HANDLERS=[]" in result.stdout
    assert "LEVEL_MATCHES=True" in result.stdout


def test_lazy_exports_cover_public_surface() -> None:
    pytest.importorskip("torch")
    result = _run_python(
        """
import worldfoundry.core.distributed as dist
missing = [name for name in dist.__all__ if not hasattr(dist, name)]
print("MISSING=" + repr(missing))
"""
    )
    _assert_clean(result)
    assert "MISSING=[]" in result.stdout, (
        "every name in worldfoundry.core.distributed.__all__ must resolve through the PEP 562 lazy export map"
    )
