"""Shared secret-marker vocabulary for redaction (infra plan SE-06).

Key matching uses whole word segments so ``max_new_tokens`` / ``tokenizer`` stay
visible while ``api_key`` / ``hf_token`` redact. Value scrubbing adds common
cloud token prefixes (GitHub, Slack, AWS).
"""

from __future__ import annotations

import re

# Whole segments that mark a key/field as sensitive.
SENSITIVE_KEY_SEGMENTS: frozenset[str] = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "passwd",
        "password",
        "pwd",
        "secret",
        "secrets",
        "token",
    }
)

# Adjacent segment pairs (individual words too generic alone).
SENSITIVE_KEY_SEGMENT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("access", "key"),
        ("api", "key"),
        ("private", "key"),
        ("secret", "key"),
        ("session", "key"),
        ("ssh", "key"),
    }
)

# Substring markers used for environment *variable names* in presence-only
# manifests (legacy env.py path). Prefer segment matching for structured keys.
SENSITIVE_ENV_SUBSTRING_MARKERS: tuple[str, ...] = (
    "TOKEN",
    "API_KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "ACCESS_KEY",
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# OpenAI/HF style plus GitHub PATs, Slack bot tokens, AWS access key ids,
# and compact JWT payloads (header.payload.signature, header starts with eyJ).
KNOWN_SECRET_VALUE_RE = re.compile(
    r"\b(?:"
    r"sk[_-][A-Za-z0-9_-]{8,}"
    r"|hf[_-][A-Za-z0-9_-]{8,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r")\b"
)


def sensitive_key_segments(key: str) -> tuple[str, ...]:
    """Split *key* into lower-case word segments for sensitive matching."""

    return tuple(
        segment
        for segment in _CAMEL_BOUNDARY.sub("_", key).lower().replace("-", "_").split("_")
        if segment
    )


def is_sensitive_key(key: str) -> bool:
    """Return whether *key* looks like a secret field name (segment match)."""

    segments = sensitive_key_segments(key)
    if any(segment in SENSITIVE_KEY_SEGMENTS for segment in segments):
        return True
    return any(pair in SENSITIVE_KEY_SEGMENT_PAIRS for pair in zip(segments, segments[1:]))


def is_sensitive_env_name(name: str) -> bool:
    """Return whether an environment variable name should hide its value.

    Prefer segment matching so ``TOKENIZER_PATH`` / ``MAX_NEW_TOKENS`` stay
    visible as presence metadata. Fall back to a short list of multi-word
    substring markers that segments alone can miss (e.g. ``API_KEY`` glued).
    """

    if is_sensitive_key(name):
        return True
    upper = name.upper()
    for marker in ("API_KEY", "SECRET", "PASSWORD", "CREDENTIAL", "ACCESS_KEY"):
        if marker in upper:
            return True
    return False
