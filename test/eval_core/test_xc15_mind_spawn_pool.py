"""XC-15: mind runner must create its multiprocessing Pool from a spawn context.

Refs plan/code_review/12_cross_cutting.md [XC-15] — the vendored mind metric
process forks Pool workers that use CUDA. ``mp.set_start_method('spawn')`` only
sets the *global* default, which another library can re-set before the Pool is
created; ``mp.get_context('spawn').Pool(...)`` pins the start method for these
workers regardless. This is a source-level check so it never imports the heavy
mind CUDA runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MIND_PROCESS_PY = (
    REPO_ROOT
    / "worldfoundry/evaluation/tasks/execution/runners/mind/runtime/mind/src/process.py"
)


def _parse() -> ast.Module:
    return ast.parse(MIND_PROCESS_PY.read_text(encoding="utf-8"), filename=str(MIND_PROCESS_PY))


def _is_spawn_get_context_call(node: ast.expr) -> bool:
    """True for ``<anything>.get_context('spawn')`` or ``get_context('spawn')``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name != "get_context":
        return False
    return bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == "spawn"


def test_xc15_pool_created_via_spawn_context() -> None:
    pool_calls: list[ast.Call] = []
    for node in ast.walk(_parse()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Pool"
        ):
            pool_calls.append(node)

    assert pool_calls, f"expected at least one Pool(...) call in {MIND_PROCESS_PY}"

    violations = [
        f"line {call.lineno}: Pool not created via get_context('spawn')"
        for call in pool_calls
        if not _is_spawn_get_context_call(call.func.value)
    ]
    assert not violations, ";\n".join(violations)


def test_xc15_global_spawn_start_method_retained() -> None:
    """Belt-and-suspenders: keep set_start_method('spawn') for mp.Manager() etc."""
    for node in ast.walk(_parse()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_start_method"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "spawn"
        ):
            return
    raise AssertionError(f"set_start_method('spawn') not found in {MIND_PROCESS_PY}")
