from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.reporting import (
    ENVIRONMENT_SCHEMA_VERSION,
    ENV_REQUIREMENTS_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    redact_secrets,
    validate_contract_file,
    write_run_manifest_artifacts,
)


def test_write_run_manifest_artifacts_redacts_secrets_and_validates(tmp_path: Path) -> None:
    paths = write_run_manifest_artifacts(
        output_dir=tmp_path,
        base_manifest={
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": "run-1",
            "runner": "unit",
            "status": "succeeded",
            "output_dir": str(tmp_path),
            "model": {"model_id": "model-a", "revision": "model-rev"},
            "dataset": {"dataset_id": "dataset-a", "revision": "dataset-rev"},
            "artifacts": {},
        },
        config={"api_key": "should-not-appear", "temperature": 0.0},
        required_env=("OPENAI_API_KEY",),
        required_paths=(tmp_path,),
        cache_paths={"hf_cache": tmp_path / "hf"},
        package_names=("worldfoundry",),
        environ={},
    )

    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    environment = json.loads(paths["environment"].read_text(encoding="utf-8"))
    env_requirements = json.loads(paths["env_requirements"].read_text(encoding="utf-8"))
    manifest_text = paths["run_manifest"].read_text(encoding="utf-8")

    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["environment"]["schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    assert manifest["env_requirements"]["schema_version"] == ENV_REQUIREMENTS_SCHEMA_VERSION
    assert "preflight" not in manifest
    assert manifest["env_requirements"]["missing_env"] == ["OPENAI_API_KEY"]
    assert manifest["config"]["api_key"] == "<redacted>"
    assert manifest["model_revision"] == "model-rev"
    assert manifest["dataset_revision"] == "dataset-rev"
    assert manifest["artifacts"]["environment"] == str(paths["environment"])
    assert manifest["artifacts"]["env_requirements"] == str(paths["env_requirements"])
    assert environment["python"]["version"]
    assert env_requirements["required_env"] == [{"name": "OPENAI_API_KEY", "present": False, "redacted": True}]
    assert "should-not-appear" not in manifest_text
    assert validate_contract_file(paths["run_manifest"], kind="run-manifest")["ok"] is True


def test_redact_secrets_recurses_nested_config() -> None:
    redacted = redact_secrets(
        {
            "safe": "visible",
            "nested": {
                "client_secret": "hidden",
                "items": [{"auth_token": "hidden-too"}],
            },
        }
    )

    assert redacted == {
        "safe": "visible",
        "nested": {
            "client_secret": "<redacted>",
            "items": [{"auth_token": "<redacted>"}],
        },
    }
