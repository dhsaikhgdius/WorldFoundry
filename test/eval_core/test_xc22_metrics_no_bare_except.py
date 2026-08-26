"""XC-22: metric modules must not use bare ``except:`` handlers.

Refs plan/code_review/12_cross_cutting.md [XC-22] — bare handlers catch
``KeyboardInterrupt`` / ``SystemExit`` and can swallow user interrupts on the
host-side metric path.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

XC22_METRIC_FILES = (
    REPO_ROOT / "worldfoundry/evaluation/tasks/metrics/jedi/utils.py",
    REPO_ROOT / "worldfoundry/evaluation/tasks/metrics/facescore/facescore_pkg/FaceScore.py",
    REPO_ROOT / "worldfoundry/evaluation/tasks/metrics/artscore/models.py",
    REPO_ROOT / "worldfoundry/evaluation/tasks/metrics/artscore/datasets.py",
    REPO_ROOT / "worldfoundry/evaluation/tasks/metrics/artscore/utils.py",
)


def _bare_except_handlers(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            lines.append(getattr(node, "lineno", 0))
    return lines


def test_xc22_metric_modules_have_no_bare_except() -> None:
    violations: list[str] = []
    for path in XC22_METRIC_FILES:
        bare = _bare_except_handlers(path)
        if bare:
            rel = path.relative_to(REPO_ROOT)
            violations.append(f"{rel}: bare except at lines {bare}")
    assert not violations, ";\n".join(violations)
