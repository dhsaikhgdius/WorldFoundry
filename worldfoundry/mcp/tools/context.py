"""Shared execution context for WorldFoundry MCP tool payloads."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR, REPO_ROOT
from worldfoundry.runtime.jobs import AsyncCommandJobStore

# ── Defaults ───────────────────────────────────────────────────


def _default_mcp_output_root() -> Path:
    """Resolve the MCP run root to a stable absolute path.

    MCP stdio servers are spawned by clients with an arbitrary working
    directory (often ``/``), so a CWD-relative default would scatter run
    outputs in unpredictable places (CM-29). ``WORLDFOUNDRY_MCP_RUN_ROOT``
    still overrides; a relative override is resolved against the CWD once,
    at import time, so it stays stable for the server's lifetime.
    """

    override = os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "runs" / "mcp"


# NOTE: ``DEFAULT_MCP_OUTPUT_ROOT`` can be overridden via the
# ``WORLDFOUNDRY_MCP_RUN_ROOT`` environment variable.
DEFAULT_MCP_OUTPUT_ROOT = _default_mcp_output_root()


@dataclass
class MCPToolContext:
    """Manifest roots, output root, and job store shared by MCP tool payloads.

    Attributes:
        output_root: Directory where MCP-triggered evaluation runs write
            their outputs. Defaults to ``DEFAULT_MCP_OUTPUT_ROOT``.
        model_manifest_dir: Root directory of the model manifest zoo, or
            ``None`` if not configured.
        benchmark_manifest_dir: Root directory of the benchmark manifest zoo.
        job_store: :class:`AsyncCommandJobStore` used to track active and
            completed evaluation runs.
    """

    output_root: Path = DEFAULT_MCP_OUTPUT_ROOT
    model_manifest_dir: Path | None = MODEL_ZOO_DIR
    benchmark_manifest_dir: Path = BENCHMARK_ZOO_DIR
    job_store: AsyncCommandJobStore = field(default_factory=AsyncCommandJobStore)


DEFAULT_CONTEXT = MCPToolContext()


__all__ = ["DEFAULT_CONTEXT", "DEFAULT_MCP_OUTPUT_ROOT", "MCPToolContext"]
