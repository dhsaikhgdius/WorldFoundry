"""Contract: eval-core CI records worldfoundry/evaluation coverage (C-06)."""

from __future__ import annotations

from pathlib import Path

CI_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def test_eval_core_job_writes_evaluation_coverage_summary() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    assert "--cov=worldfoundry/evaluation" in text
    assert "Write evaluation coverage to job summary" in text
    assert "coverage report --format=total" in text
    assert "GITHUB_STEP_SUMMARY" in text
    # Do not regress tip SHA pins while adding coverage.
    assert "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09" in text
