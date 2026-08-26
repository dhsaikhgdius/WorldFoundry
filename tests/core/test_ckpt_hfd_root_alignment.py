"""DS-02: CKPT/HFD defaults align with unified_install + link_hf."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import hfd_root_path, worldfoundry_path_tokens
from worldfoundry.runtime.env import SOURCEABLE_ENV_BASE_LINES


def test_ckpt_default_under_model_dir(tmp_path: Path) -> None:
    tokens = worldfoundry_path_tokens({"WORLDFOUNDRY_HOME": str(tmp_path / "home")})
    assert tokens["WORLDFOUNDRY_CKPT_DIR"] == str(tmp_path / "home" / "models" / "checkpoints")
    assert tokens["WORLDFOUNDRY_HFD_ROOT"] == str(Path(tokens["WORLDFOUNDRY_CKPT_DIR"]) / "hfd")
    assert hfd_root_path(env={"WORLDFOUNDRY_HOME": str(tmp_path / "home")}) == Path(
        tokens["WORLDFOUNDRY_HFD_ROOT"]
    )


def test_sourceable_env_ckpt_uses_model_dir() -> None:
    joined = "\n".join(SOURCEABLE_ENV_BASE_LINES)
    assert 'WORLDFOUNDRY_CKPT_DIR="${WORLDFOUNDRY_CKPT_DIR:-${WORLDFOUNDRY_MODEL_DIR}/checkpoints}"' in joined


def test_unified_install_live_exports_ckpt_dir() -> None:
    script = Path("scripts/setup/unified_install.sh").read_text(encoding="utf-8")
    # Live process exports (outside the WRITE_ENV_FILE heredoc) must include CKPT_DIR.
    live_marker = 'export WORLDFOUNDRY_CKPT_DIR="${MODEL_ROOT}/checkpoints"'
    assert script.count(live_marker) >= 2  # env file + live shell exports
    assert "export WORLDFOUNDRY_BENCHMARK_DATA_ROOT=" in script


def test_download_checkpoints_default_uses_hfd_root_path() -> None:
    text = Path("scripts/model_zoo/download_checkpoints.py").read_text(encoding="utf-8")
    assert "DEFAULT_CACHE_DIR = hfd_root_path()" in text
    assert 'REPO_ROOT / "cache" / "hfd"' not in text
