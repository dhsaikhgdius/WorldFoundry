"""XC-20: library code must not mutate the process-wide root logger.

Guards the fixes from plan/code_review/12_cross_cutting.md [XC-20]: importing a
metric module or calling a core helper must never reconfigure the host
process's root logger (handlers or level). Only the central
``worldfoundry.core.logging_setup`` and explicit CLI entry points may touch it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories whose modules are imported into the host (orchestrator) process.
# Vendored runner runtimes (runners/*/runtime) run as subprocesses and are out
# of scope here.
HOST_SIDE_DIRS = (
    "worldfoundry/core",
    "worldfoundry/cli",
    "worldfoundry/mcp",
    "worldfoundry/pipelines",
    "worldfoundry/studio",
    "worldfoundry/training",
    "worldfoundry/runtime",
    "worldfoundry/operators",
    "worldfoundry/evaluation/tasks/metrics",
)

# The central logging config is the single owner of the root logger.
WHITELIST = frozenset({
    "worldfoundry/core/logging_setup.py",
})

FORBIDDEN_PATTERNS = (
    re.compile(r"logging\.basicConfig\("),
    re.compile(r"logging\.disable\("),
    re.compile(r"logging\.getLogger\(\)\.(setLevel|addHandler)"),
    # Bare root-logger acquisition assigned to a name (the pre-fix V_JEPA
    # pattern: ``logger = logging.getLogger()`` then ``logger.setLevel``).
    re.compile(r"=\s*logging\.getLogger\(\)\s*$"),
)


def _violations() -> list[str]:
    found: list[str] = []
    for rel_dir in HOST_SIDE_DIRS:
        for path in sorted((REPO_ROOT / rel_dir).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in WHITELIST:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if line.lstrip().startswith("#"):
                    continue
                for pattern in FORBIDDEN_PATTERNS:
                    if pattern.search(line):
                        found.append(f"{rel}:{lineno}: {line.strip()}")
    return found


def test_host_side_packages_do_not_touch_root_logger() -> None:
    violations = _violations()
    assert not violations, (
        "Library code must not reconfigure the root logger (XC-20). Use "
        "logging.getLogger(__name__) or worldfoundry.core.logging_setup. "
        "Offending lines:\n" + "\n".join(violations)
    )


def test_jedi_vjepa_uses_module_private_logger() -> None:
    text = (
        REPO_ROOT / "worldfoundry/evaluation/tasks/metrics/jedi/V_JEPA.py"
    ).read_text(encoding="utf-8")
    assert "logger = logging.getLogger(__name__)" in text
    assert "logger.setLevel" not in text


def test_wan_video_geometry_uses_module_private_logger() -> None:
    text = (REPO_ROOT / "worldfoundry/core/io/wan_video_geometry.py").read_text(
        encoding="utf-8"
    )
    assert "logger = logging.getLogger(__name__)" in text
    # All log calls go through the module logger, never the root shortcuts.
    assert not re.search(r"^\s*logging\.(info|error|warning|debug)\(", text, re.M)


def test_memobench_run_eval_defers_root_config_to_main() -> None:
    text = (
        REPO_ROOT
        / "worldfoundry/evaluation/tasks/execution/runners/memobench/runtime"
        / "memobench/evaluation/run_eval.py"
    ).read_text(encoding="utf-8")
    prelude, _, body = text.partition("def main(")
    assert body, "run_eval.py must keep a main() entry point"
    # Importing the module must not silence the host process...
    assert "logging.getLogger().setLevel" not in prelude
    assert "logging.disable(" not in prelude
    assert "logging.basicConfig(" not in prelude
    # ...but the CLI entry point keeps the intended spam suppression.
    assert "logging.getLogger().setLevel(logging.WARNING)" in body
    assert "logging.disable(logging.INFO)" in body
