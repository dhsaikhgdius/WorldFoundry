from __future__ import annotations

from worldfoundry.core.logging_setup import redact_sensitive_text
from worldfoundry.core.secret_patterns import is_sensitive_env_name, is_sensitive_key
from worldfoundry.evaluation.reporting.run_manifest import _is_sensitive_key
from worldfoundry.runtime.env import _is_sensitive_key as env_is_sensitive_key


def test_secret_key_vocab_is_shared_across_manifest_and_env() -> None:
    assert _is_sensitive_key("hf_token") is True
    assert env_is_sensitive_key("HF_TOKEN") is True
    assert is_sensitive_key("max_new_tokens") is False
    assert is_sensitive_env_name("TOKENIZER_PATH") is False


def test_known_secret_value_patterns_cover_cloud_prefixes() -> None:
    samples = (
        "ghp_abcdefghijklmnopqrstuv",
        "github_pat_abcdefghijklmnopqrstuvwxyz",
        "xoxb-1234567890-abcdefghij",
        "AKIAIOSFODNN7EXAMPLE",
        "sk-abcdefghijklmnop",
        "hf_abcdefghijklmnop",
    )
    for sample in samples:
        redacted = redact_sensitive_text(f"value={sample}")
        assert sample not in redacted, sample
        assert "[REDACTED]" in redacted
