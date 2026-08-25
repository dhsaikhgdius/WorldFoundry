#!/usr/bin/env python3
"""Split workspace_registry.py into a package."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners/workspace_registry.py"
BASE = SRC.parent / "workspace_registry"


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
    specs_body = "".join(lines[21:491])
    dispatch_body = "".join(lines[491:])

    specs = (
        '"""Workspace runner specifications for Studio dispatch."""\n\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n"
        + specs_body
    )
    dispatch = (
        '"""Workspace benchmark dispatch helpers."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "from dataclasses import asdict\n"
        "from functools import lru_cache\n"
        "from pathlib import Path\n"
        "from typing import Any, Callable, Mapping, Sequence\n\n"
        "from worldfoundry.core.io.paths import project_root\n"
        "from worldfoundry.core.process import read_text_tail, run_logged_subprocess\n"
        "from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset\n\n"
        "from .specs import CLI_RUNNERS, GENERIC_EVALUATION_METRICS, RESULT_SUFFIXES, WorkspaceRunnerSpec\n\n"
        "REPO_ROOT = project_root(__file__)\n\n"
        + dispatch_body
    )
    init = (
        '"""Workspace-facing dispatch for in-tree benchmark runners."""\n\n'
        "from .dispatch import *\n"
        "from .specs import CLI_RUNNERS, GENERIC_EVALUATION_METRICS, RESULT_SUFFIXES, WorkspaceRunnerSpec\n\n"
        "__all__ = [name for name in globals() if not name.startswith('_')]\n"
    )

    BASE.mkdir(exist_ok=True)
    (BASE / "specs.py").write_text(specs, encoding="utf-8")
    (BASE / "dispatch.py").write_text(dispatch, encoding="utf-8")
    (BASE / "__init__.py").write_text(init, encoding="utf-8")
    SRC.unlink()
    print("split workspace_registry -> package")


if __name__ == "__main__":
    main()
