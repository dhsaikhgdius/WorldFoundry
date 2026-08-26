"""C-09: bare pytest collection guards when optional deps are missing."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CONFTEST = REPO_ROOT / "test" / "conftest.py"
TESTS_CONFTEST = REPO_ROOT / "tests" / "conftest.py"
LINGBOT = REPO_ROOT / "test" / "test_lingbot_runtime_paths.py"


def test_c09_test_conftest_skips_torch_heavy_without_torch() -> None:
    text = TEST_CONFTEST.read_text(encoding="utf-8")
    assert "def pytest_ignore_collect" in text
    assert "eval_core" in text
    assert "test_lingbot_runtime_paths.py" in text
    assert "_torch_usable" in text


def test_c09_tests_conftest_skips_torch_subtrees() -> None:
    text = TESTS_CONFTEST.read_text(encoding="utf-8")
    assert "def pytest_ignore_collect" in text
    assert "synthesis/" in text
    assert "training/" in text


def test_c09_lingbot_paths_importorskip_torch() -> None:
    text = LINGBOT.read_text(encoding="utf-8")
    assert 'pytest.importorskip("torch")' in text
