from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.models.runtime.assets import (
    resolve_profile_checkpoint,
    resolve_profile_checkpoints,
)
from worldfoundry.evaluation.models.runtime.profiles import RuntimeProfile


def test_resolve_profile_checkpoints_expands_local_and_hf():
    profile = RuntimeProfile(
        model_id="demo-prior",
        display_name="Demo",
        task_family="three_dimension",
        checkpoints=(
            {"local_dir": "${WORLDFOUNDRY_CKPT_DIR}/DemoWeights"},
            {"repo_id": "org/demo", "filename": "model.pth", "role": "weights"},
        ),
    )

    paths = resolve_profile_checkpoints(profile)
    assert any(path.endswith("DemoWeights") or "DemoWeights" in path for path in paths)
    assert "hf://org/demo/model.pth" in paths


def test_resolve_profile_checkpoint_prefers_existing_local(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    ready = tmp_path / "Ready"
    ready.mkdir()
    (ready / "w.bin").write_bytes(b"ok")
    profile = RuntimeProfile(
        model_id="demo-prior",
        display_name="Demo",
        task_family="three_dimension",
        checkpoints=(
            {"local_dir": str(tmp_path / "Missing")},
            {"local_dir": str(ready)},
            {"repo_id": "org/demo"},
        ),
    )

    assert resolve_profile_checkpoint(profile) == str(ready)


def test_resolve_profile_checkpoint_falls_back_to_uri_when_nothing_local(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    profile = RuntimeProfile(
        model_id="demo-prior",
        display_name="Demo",
        task_family="three_dimension",
        checkpoints=({"repo_id": "org/demo", "filename": "model.pth"},),
    )

    assert resolve_profile_checkpoint(profile) == "hf://org/demo/model.pth"


def test_geometry_priors_checkpoint_root_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    from worldfoundry.synthesis.visual_generation import geometry_priors as gp

    assert gp._checkpoint_root() == Path(tmp_path)


def test_check_no_parents_ckpt_clean_on_repo():
    from scripts.ci.check_no_parents_ckpt import find_parents_ckpt_hits

    assert find_parents_ckpt_hits() == []
