from __future__ import annotations

from pathlib import Path

from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime


def test_official_video_runtime_checks_nested_required_paths(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "runtime"
    checkpoint_root = tmp_path / "ckpt"
    repo_root.mkdir()
    checkpoint_root.mkdir()

    config = {
        "runtime": {
            "kind": "official_cli",
            "repo_root_candidates": [str(repo_root)],
            "checkpoint_candidates": [str(checkpoint_root)],
            "required_paths": [
                {"id": "text_encoder", "path": "text_encoder/llm/config.json"},
                {"id": "entrypoint", "base": "repo", "path": "generate.py"},
            ],
            "command": ["{python}", "{repo_root}/generate.py", "--model_path", "{checkpoint_path}"],
        }
    }
    monkeypatch.setattr(OfficialVideoRuntime, "_load_config", staticmethod(lambda _: config))

    runtime = OfficialVideoRuntime(model_id="fixture", runtime_config_path="unused.yaml")
    missing_plan = runtime.runtime_plan(output_path=tmp_path / "out.mp4", prompt="demo")

    assert missing_plan["ready"] is False
    assert any("text_encoder" in item for item in missing_plan["missing"])
    assert any("entrypoint" in item for item in missing_plan["missing"])

    (checkpoint_root / "text_encoder" / "llm").mkdir(parents=True)
    (checkpoint_root / "text_encoder" / "llm" / "config.json").write_text("{}", encoding="utf-8")
    (repo_root / "generate.py").write_text("print('ok')\n", encoding="utf-8")

    ready_plan = runtime.runtime_plan(output_path=tmp_path / "out.mp4", prompt="demo")

    assert ready_plan["ready"] is True
    assert ready_plan["missing"] == []


def test_materialize_cli_artifacts_preserves_media_suffixes(tmp_path: Path) -> None:
    produced_audio = tmp_path / "official.flac"
    produced_video = tmp_path / "official.mp4"
    produced_audio.write_bytes(b"fLaC-audio")
    produced_video.write_bytes(b"\x00\x00\x00\x18ftyp-video")

    output_path = tmp_path / "workspace.mp4"
    primary, artifacts = OfficialVideoRuntime._materialize_cli_artifacts(
        produced_audio,
        output_path,
        since=0.0,
    )

    assert primary == tmp_path / "workspace.flac"
    assert primary.read_bytes() == produced_audio.read_bytes()
    assert output_path.read_bytes() == produced_video.read_bytes()
    assert artifacts == (tmp_path / "workspace.flac", output_path)
