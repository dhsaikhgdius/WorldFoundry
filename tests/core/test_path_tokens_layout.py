from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import worldfoundry_path_tokens


def test_default_ckpt_dir_is_under_models_checkpoints(tmp_path: Path, monkeypatch):
    home = tmp_path / "wf-home"
    monkeypatch.delenv("WORLDFOUNDRY_CKPT_DIR", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_HFD_ROOT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_MODEL_DIR", raising=False)
    tokens = worldfoundry_path_tokens({"WORLDFOUNDRY_HOME": str(home)})
    assert Path(tokens["WORLDFOUNDRY_CKPT_DIR"]) == home / "models" / "checkpoints"
    assert Path(tokens["WORLDFOUNDRY_HFD_ROOT"]) == home / "models" / "checkpoints" / "hfd"


def test_no_adjacent_ckpt_magic(tmp_path: Path, monkeypatch):
    # Even if a sibling ../ckpt directory exists, defaults stay under WORLDFOUNDRY_HOME.
    repo_parent = tmp_path / "workspace"
    fake_ckpt = repo_parent / "ckpt"
    fake_ckpt.mkdir(parents=True)
    home = tmp_path / "home"
    tokens = worldfoundry_path_tokens({"WORLDFOUNDRY_HOME": str(home)})
    assert Path(tokens["WORLDFOUNDRY_CKPT_DIR"]) == home / "models" / "checkpoints"
    assert Path(tokens["WORLDFOUNDRY_CKPT_DIR"]) != fake_ckpt
