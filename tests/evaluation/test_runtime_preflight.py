from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.tasks.execution.orchestration.runtime_preflight import (
    SCHEMA_VERSION,
    check_profile,
    run_preflight,
)


def _profile(**overrides):
    profile = {
        "id": "tiny-benchmark",
        "environment_id": "test-env",
        "python_path": sys.executable,
        "pythonpath_roots": [],
        "required_env": [],
        "required_imports": ["json"],
        "required_paths": [],
        "requires_cuda_visibility": False,
    }
    profile.update(overrides)
    return profile


def test_check_profile_accepts_real_paths_imports_and_env_alternatives(tmp_path: Path) -> None:
    asset = tmp_path / "assets" / "sample.json"
    asset.parent.mkdir()
    asset.write_text("{}", encoding="utf-8")

    report = check_profile(
        _profile(
            python_path="${TEST_PYTHON:-python}",
            required_env=["PRIMARY_INPUT or FALLBACK_INPUT"],
            required_paths=[{"id": "asset", "path": "${ASSET_ROOT:-assets}/sample.json"}],
        ),
        manifest_path=tmp_path / "profile.yaml",
        repo_root=tmp_path,
        environ={"TEST_PYTHON": sys.executable, "FALLBACK_INPUT": "configured"},
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["ok"] is True
    assert report["checks"]["required_env"][0]["present"] == ["FALLBACK_INPUT"]
    assert report["checks"]["required_paths"][0]["exists"] is True
    assert report["checks"]["required_imports"][0]["ok"] is True


def test_check_profile_fails_closed_for_every_declared_requirement(tmp_path: Path) -> None:
    report = check_profile(
        _profile(
            required_env=["MISSING_INPUT"],
            required_imports=["module_that_does_not_exist_worldfoundry_test"],
            required_paths=[{"id": "missing", "path": "missing.file"}],
            base_model_dependency_preflight={"status": "blocked", "missing": ["checkpoint"]},
        ),
        manifest_path=tmp_path / "profile.yaml",
        repo_root=tmp_path,
        environ={},
    )

    assert report["ok"] is False
    assert report["summary"] == {
        "missing_env_groups": 1,
        "missing_required_paths": 1,
        "missing_pythonpath_roots": 0,
        "failed_imports": 1,
    }
    assert report["checks"]["base_models"]["missing"] == ["checkpoint"]


def test_import_errors_are_secret_redacted(tmp_path: Path) -> None:
    secret = "do-not-leak-this-api-key"
    module = tmp_path / "fails_with_secret.py"
    module.write_text(
        "import os\nraise RuntimeError(os.environ['PRIVATE_API_KEY'])\n",
        encoding="utf-8",
    )

    report = check_profile(
        _profile(
            pythonpath_roots=[str(tmp_path)],
            required_imports=["fails_with_secret"],
        ),
        manifest_path=tmp_path / "profile.yaml",
        repo_root=tmp_path,
        environ={"PRIVATE_API_KEY": secret},
    )

    encoded = json.dumps(report)
    assert report["ok"] is False
    assert secret not in encoded
    assert "<redacted>" in encoded


def test_optional_path_does_not_make_preflight_fail(tmp_path: Path) -> None:
    report = check_profile(
        _profile(required_paths=[{"id": "optional", "path": "absent", "required_for_env": False}]),
        manifest_path=tmp_path / "profile.yaml",
        repo_root=tmp_path,
        environ={},
    )

    row = report["checks"]["required_paths"][0]
    assert report["ok"] is True
    assert row == {
        "id": "optional",
        "path": str(tmp_path / "absent"),
        "required": False,
        "source_env": None,
        "missing_expansion_env": [],
        "exists": False,
        "ok": True,
    }


def test_run_preflight_writes_report_and_rejects_wrong_profile(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny-benchmark.yaml"
    manifest.write_text(yaml.safe_dump(_profile()), encoding="utf-8")
    output_dir = tmp_path / "output"

    report = run_preflight(
        profile="tiny-benchmark",
        manifest=manifest,
        output_dir=output_dir,
        repo_root=tmp_path,
        environ={},
    )

    assert report["ok"] is True
    persisted = json.loads((output_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert persisted["profile_id"] == "tiny-benchmark"
    with pytest.raises(ValueError, match="does not match"):
        run_preflight(
            profile="different-benchmark",
            manifest=manifest,
            output_dir=tmp_path / "wrong",
            repo_root=tmp_path,
            environ={},
        )
