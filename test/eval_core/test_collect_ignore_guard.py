"""TE-04: keep ``test/conftest.py`` collect_ignore disjoint from real tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"
CONFTEST_PATH = TEST_ROOT / "conftest.py"


def _load_collect_ignore() -> list[str]:
    tree = ast.parse(CONFTEST_PATH.read_text(encoding="utf-8"), filename=str(CONFTEST_PATH))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "collect_ignore":
                value = ast.literal_eval(node.value)
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise AssertionError("collect_ignore must be a list[str]")
                return value
    raise AssertionError(f"collect_ignore not found in {CONFTEST_PATH}")


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _defines_pytest_or_unittest_tests(path: Path) -> bool:
    """Return True when the module defines collectable pytest/unittest tests."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            return True
        if not isinstance(node, ast.ClassDef):
            continue
        has_test_method = any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_")
            for child in node.body
        )
        if not has_test_method:
            continue
        bases = {_base_name(base) for base in node.bases}
        bases.discard(None)
        if node.name.startswith("Test") or "TestCase" in bases:
            return True
        # unittest-style classes often end with ``Test`` without subclassing TestCase in the AST
        # (e.g. ``class FooTest(unittest.TestCase)`` already covered via Attribute base).
        if node.name.endswith("Test"):
            return True
    return False


@pytest.mark.unit
def test_collect_ignore_entries_exist_and_are_demos() -> None:
    ignore = _load_collect_ignore()
    assert ignore, "collect_ignore must not be empty"
    assert len(ignore) == len(set(ignore)), "collect_ignore has duplicate entries"

    for name in ignore:
        path = TEST_ROOT / name
        assert path.is_file(), f"collect_ignore entry missing on disk: {name}"
        assert not _defines_pytest_or_unittest_tests(path), (
            f"{name} is in collect_ignore but defines test_* / Test* cases; "
            "remove it from collect_ignore or move demos out of test/"
        )


@pytest.mark.unit
def test_top_level_test_modules_partition_ignore_vs_real() -> None:
    """Every top-level ``test/test_*.py`` is either a demo (ignored) or a real test module."""

    ignore = set(_load_collect_ignore())
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        is_ignored = path.name in ignore
        has_tests = _defines_pytest_or_unittest_tests(path)
        if is_ignored:
            assert not has_tests, f"{path.name} ignored but contains real tests"
        else:
            assert has_tests, (
                f"{path.name} is not in collect_ignore and has no test_* / Test* definitions; "
                "add it to collect_ignore (demo) or convert it into a real test"
            )
