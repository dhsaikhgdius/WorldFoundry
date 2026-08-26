"""R-09: default env export file under WORLDFOUNDRY_HOME (tip-stack of #61)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    REPO_ROOT / "scripts" / "setup" / "bootstrap_worldfoundry.sh",
    REPO_ROOT / "scripts" / "setup" / "unified_install.sh",
    REPO_ROOT / "scripts" / "workspace" / "run_workspace.sh",
)


def test_r09_env_file_defaults_under_worldfoundry_home() -> None:
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert 'ENV_FILE="${WORLDFOUNDRY_ENV_FILE:-${WF_CACHE_HOME}/worldfoundry_unified_env.sh}"' in text
        assert "tmp/worldfoundry_unified_env.sh" not in text or "Default: tmp/" not in text
        # no bare tmp default left for ENV_FILE
        assert 'ENV_FILE="${WORLDFOUNDRY_ENV_FILE:-tmp/worldfoundry_unified_env.sh}"' not in text


def test_r09_preserves_i13_wrapper_rationale() -> None:
    text = (REPO_ROOT / "scripts" / "setup" / "unified_install.sh").read_text(encoding="utf-8")
    assert "plan I-13" in text
