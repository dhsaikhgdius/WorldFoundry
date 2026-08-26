"""R-08: default env export file lives under WORLDFOUNDRY_HOME, not repo tmp/."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPTS = (
    REPO_ROOT / "scripts" / "setup" / "bootstrap_worldfoundry.sh",
    REPO_ROOT / "scripts" / "setup" / "unified_install.sh",
    REPO_ROOT / "scripts" / "workspace" / "run_workspace.sh",
)


def test_no_repo_local_tmp_env_file_default() -> None:
    for script in SETUP_SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert "tmp/worldfoundry_unified_env.sh" not in text, script.name


def test_env_file_defaults_under_worldfoundry_home() -> None:
    for script in SETUP_SCRIPTS:
        text = script.read_text(encoding="utf-8")
        # Explicit override always wins ...
        assert "WORLDFOUNDRY_ENV_FILE" in text, script.name
        # ... and the fallback is anchored to the runtime home, not the checkout.
        assert "worldfoundry_unified_env.sh" in text, script.name
        assert "WORLDFOUNDRY_HOME" in text, script.name


def test_home_override_moves_env_file_default() -> None:
    # bootstrap/unified_install accept --home, so their env-file default must
    # be resolved after option parsing rather than frozen at script start.
    for script in SETUP_SCRIPTS[:2]:
        text = script.read_text(encoding="utf-8")
        assert '${HOME_ROOT}/worldfoundry_unified_env.sh' in text, script.name


def test_benchmark_asset_plan_bootstrap_guidance() -> None:
    text = (REPO_ROOT / "scripts" / "setup" / "prepare_benchmark_assets.py").read_text(encoding="utf-8")
    assert "tmp/worldfoundry_unified_env.sh" not in text
