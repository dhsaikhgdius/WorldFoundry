from __future__ import annotations

import json
import os

from worldfoundry.studio.conda_dispatch import (
    DISPATCH_API_KEY_ENV,
    SECRET_ENV_REF_KEY,
    _payload_run_kwargs_with_secret_refs,
)
from worldfoundry.studio.runtime_job import _resolve_secret_refs


def test_payload_lifts_nested_secrets_out_of_kwargs_text() -> None:
    payload, secret_env = _payload_run_kwargs_with_secret_refs(
        {
            "api_key": "top-secret",
            "load_kwargs_text": json.dumps({"hf_token": "nested-hf", "temperature": 0.2}),
            "call_kwargs_text": json.dumps({"nested": {"api_key": "deep-secret"}}),
        }
    )

    assert payload["api_key"] == {SECRET_ENV_REF_KEY: DISPATCH_API_KEY_ENV}
    assert secret_env[DISPATCH_API_KEY_ENV] == "top-secret"
    assert "nested-hf" not in payload["load_kwargs_text"]
    assert "deep-secret" not in payload["call_kwargs_text"]

    for key, value in secret_env.items():
        os.environ[key] = value
    try:
        resolved = _resolve_secret_refs(payload)
        assert resolved["api_key"] == "top-secret"
        assert json.loads(resolved["load_kwargs_text"])["hf_token"] == "nested-hf"
        assert json.loads(resolved["call_kwargs_text"])["nested"]["api_key"] == "deep-secret"
    finally:
        for key in secret_env:
            os.environ.pop(key, None)


def test_grounding_dino_init_distributed_does_not_dump_raw_environ() -> None:
    text = (
        __import__("pathlib")
        .Path("worldfoundry/base_models/perception_core/detection/grounding_dino/util/misc.py")
        .read_text(encoding="utf-8")
    )
    assert "json.dumps(dict(os.environ)" not in text
    assert "redact_env_for_manifest" in text
