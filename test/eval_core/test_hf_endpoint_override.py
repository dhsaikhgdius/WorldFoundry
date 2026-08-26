"""ST-01: Studio child processes must not default to a third-party HF mirror.

``worldfoundry.studio.conda_dispatch._runtime_env`` used to
``setdefault("HF_ENDPOINT", "https://hf-mirror.com")``, silently routing all
child-process Hugging Face traffic through a third-party mirror. Mirrors are
now strictly opt-in via ``WORLDFOUNDRY_HF_ENDPOINT``.

The dispatch module pulls in heavy Studio dependencies, so the wiring is
checked as a source contract while the helper itself is unit-tested through
the lightweight ``worldfoundry.runtime.env`` module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from worldfoundry.runtime.env import HF_ENDPOINT_OVERRIDE_ENV, apply_hf_endpoint_override


def _conda_dispatch_source() -> str:
    spec = importlib.util.find_spec("worldfoundry.studio.conda_dispatch")
    assert spec is not None and spec.origin, "worldfoundry.studio.conda_dispatch must be locatable"
    return Path(spec.origin).read_text(encoding="utf-8")


def test_override_env_name() -> None:
    assert HF_ENDPOINT_OVERRIDE_ENV == "WORLDFOUNDRY_HF_ENDPOINT"


def test_no_endpoint_by_default() -> None:
    env: dict[str, str] = {}
    assert apply_hf_endpoint_override(env) is None
    assert "HF_ENDPOINT" not in env


def test_inherited_endpoint_is_preserved() -> None:
    env = {"HF_ENDPOINT": "https://hf-mirror.example"}
    assert apply_hf_endpoint_override(env) == "https://hf-mirror.example"
    assert env["HF_ENDPOINT"] == "https://hf-mirror.example"


def test_opt_in_override_sets_endpoint() -> None:
    env = {HF_ENDPOINT_OVERRIDE_ENV: "https://hf-mirror.example"}
    assert apply_hf_endpoint_override(env) == "https://hf-mirror.example"
    assert env["HF_ENDPOINT"] == "https://hf-mirror.example"


def test_opt_in_override_wins_over_inherited_endpoint() -> None:
    env = {
        HF_ENDPOINT_OVERRIDE_ENV: "https://mirror.example",
        "HF_ENDPOINT": "https://other.example",
    }
    assert apply_hf_endpoint_override(env) == "https://mirror.example"
    assert env["HF_ENDPOINT"] == "https://mirror.example"


def test_blank_override_is_ignored() -> None:
    env = {HF_ENDPOINT_OVERRIDE_ENV: "   "}
    assert apply_hf_endpoint_override(env) is None
    assert "HF_ENDPOINT" not in env


def test_conda_dispatch_has_no_default_mirror() -> None:
    source = _conda_dispatch_source()
    assert "hf-mirror.com" not in source, (
        "conda_dispatch must not hardcode a third-party HF mirror; "
        "mirrors are opt-in via WORLDFOUNDRY_HF_ENDPOINT (ST-01)"
    )
    assert "apply_hf_endpoint_override(env)" in source, (
        "conda_dispatch._runtime_env must route HF_ENDPOINT through "
        "worldfoundry.runtime.env.apply_hf_endpoint_override (ST-01)"
    )
