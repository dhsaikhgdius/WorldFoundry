"""DA-01 regression: tracked test_cases must not grow past the pinned baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_test_cases_growth.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_test_cases_growth", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_test_cases_growth_within_baseline() -> None:
    module = _load_module()
    file_count, total_bytes = module.measure_tracked_test_cases(REPO_ROOT)
    assert file_count <= module.MAX_TRACKED_FILES
    assert total_bytes <= module.MAX_TRACKED_BYTES
    assert module.main([]) == 0


def test_test_cases_growth_script_fails_when_limits_tightened() -> None:
    module = _load_module()
    file_count, total_bytes = module.measure_tracked_test_cases(REPO_ROOT)
    assert file_count >= 1
    assert module.main(["--max-files", "0"]) == 1
    assert module.main(["--max-bytes", "0"]) == 1
    # Sanity: current tree still fits the documented ceilings.
    assert file_count <= 500
    assert total_bytes <= 338 * 1024 * 1024
