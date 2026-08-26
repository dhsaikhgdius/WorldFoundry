"""XC-09: worldolympiad OpenRouter key must use conventional env names.

Refs plan/code_review/12_cross_cutting.md [XC-9] — lowercase ``api_key`` is
non-standard and collision-prone; prefer OPENROUTER_API_KEY with legacy fallback.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENROUTER_MODULE = (
    REPO_ROOT
    / "worldfoundry/evaluation/tasks/execution/runners/worldolympiad/runtime/worldolympiad/model/openrouter.py"
)


def _load_openrouter_module():
    spec = importlib.util.spec_from_file_location("worldolympiad_openrouter_under_test", OPENROUTER_MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_xc09_openrouter_api_key_prefers_conventional_env_names(monkeypatch) -> None:
    module = _load_openrouter_module()
    for name in ("OPENROUTER_API_KEY", "WORLDFOUNDRY_OPENROUTER_API_KEY", "api_key"):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("WORLDFOUNDRY_OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("api_key", raising=False)

    monkeypatch.setenv("OPENROUTER_API_KEY", "primary-key")
    monkeypatch.setenv("WORLDFOUNDRY_OPENROUTER_API_KEY", "secondary-key")
    monkeypatch.setenv("api_key", "legacy-key")
    assert module.openrouter_api_key() == "primary-key"

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert module.openrouter_api_key() == "secondary-key"

    monkeypatch.delenv("WORLDFOUNDRY_OPENROUTER_API_KEY", raising=False)
    assert module.openrouter_api_key() == "legacy-key"

    monkeypatch.delenv("api_key", raising=False)
    assert module.openrouter_api_key() is None


def test_xc09_openrouter_module_does_not_read_legacy_api_key_directly() -> None:
    text = OPENROUTER_MODULE.read_text(encoding="utf-8")
    assert "getenv('api_key')" not in text
    assert 'getenv("api_key")' not in text
