"""AP-02/AP-03: schema_version -v1 aliases + reject NaN in from_json."""

from __future__ import annotations

import math
import pytest

from worldfoundry.evaluation.api.artifacts import ARTIFACT_REF_SCHEMA_VERSION, ArtifactRef
from worldfoundry.evaluation.api.generation import GENERATION_RESULT_SCHEMA_VERSION, GenerationResult
from worldfoundry.evaluation.api.json_contract import (
    json_sha256,
    require_schema_version,
    stable_hash_data,
    supported_schema_versions,
)


def test_supported_schema_versions_includes_legacy_alias() -> None:
    supported = supported_schema_versions("worldfoundry-artifact-ref-v1")
    assert "worldfoundry-artifact-ref-v1" in supported
    assert "worldfoundry-artifact-ref" in supported


def test_require_schema_version_normalizes_legacy() -> None:
    assert (
        require_schema_version(
            "worldfoundry-generation-result",
            current=GENERATION_RESULT_SCHEMA_VERSION,
            label="GenerationResult",
        )
        == GENERATION_RESULT_SCHEMA_VERSION
    )


def test_artifact_ref_accepts_legacy_schema_version() -> None:
    ref = ArtifactRef(uri="memory://x", kind="video", schema_version="worldfoundry-artifact-ref")
    assert ref.schema_version == ARTIFACT_REF_SCHEMA_VERSION


def test_from_json_rejects_nan() -> None:
    payload = '{"uri":"memory://x","kind":"video","schema_version":"worldfoundry-artifact-ref-v1","size_bytes":NaN}'
    with pytest.raises(ValueError, match="non-finite"):
        ArtifactRef.from_json(payload)


def test_stable_hash_data_matches_json_sha256() -> None:
    data = {"b": 2, "a": 1, "nested": {"z": True, "y": [1, 2]}}
    assert stable_hash_data(data) == json_sha256(data)


def test_generation_result_emits_v1() -> None:
    result = GenerationResult(sample_id="s1")
    assert result.schema_version == "worldfoundry-generation-result-v1"
