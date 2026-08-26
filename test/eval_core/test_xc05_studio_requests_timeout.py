"""XC-05: studio plugin downloads must bound ``requests.get`` with a timeout.

Refs plan/code_review/12_cross_cutting.md [XC-5] — unbounded HTTP calls can hang
the Studio host process when a remote endpoint stalls.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Self-authored studio paths called out in XC-5 (artifacts.py already has timeout).
XC05_STUDIO_REQUEST_GET_FILES = (
    REPO_ROOT / "worldfoundry/studio/visualization/plugins/perception/sky_segmentation.py",
    REPO_ROOT / "worldfoundry/studio/visualization/plugins/scene3d/glb_export.py",
)


def _requests_get_calls_missing_timeout(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id == "requests"
        ):
            continue
        kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        if "timeout" not in kw_names:
            missing.append(getattr(node, "lineno", 0))
    return missing


def test_xc05_studio_plugin_requests_get_always_pass_timeout() -> None:
    violations: list[str] = []
    for path in XC05_STUDIO_REQUEST_GET_FILES:
        missing = _requests_get_calls_missing_timeout(path)
        if missing:
            rel = path.relative_to(REPO_ROOT)
            violations.append(f"{rel}: requests.get without timeout at lines {missing}")
    assert not violations, ";\n".join(violations)
