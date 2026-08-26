from __future__ import annotations

import subprocess
from pathlib import Path


def test_scope_predict_uses_run_logged_subprocess(monkeypatch, tmp_path):
    from worldfoundry.synthesis.visual_generation.scope import worldfoundry_runtime as scope_rt

    calls: list[dict] = []

    def fake_logged(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(scope_rt, "run_logged_subprocess", fake_logged)
    monkeypatch.setattr(scope_rt, "synthesis_timeout_seconds", lambda default=None: 12.0)
    monkeypatch.setattr(scope_rt, "runtime_root", lambda: tmp_path)

    action = tmp_path / "a.parquet"
    action.write_bytes(b"a")
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    out = tmp_path / "out" / "video.mp4"
    out.parent.mkdir(parents=True)

    def fake_work_dir(self, output_path):
        work = tmp_path / "work"
        (work / "outputs").mkdir(parents=True, exist_ok=True)
        (work / "outputs" / "clip.mp4").write_bytes(b"mp4")
        return work

    monkeypatch.setattr(scope_rt.SCOPERuntime, "_work_dir", fake_work_dir)
    monkeypatch.setattr(
        scope_rt.SCOPERuntime,
        "_materialize_image",
        lambda self, images, work_dir: image,
    )
    monkeypatch.setattr(
        scope_rt.SCOPERuntime,
        "_resolve_action_path",
        lambda self, action_path=None, interactions=(): action,
    )
    monkeypatch.setattr(
        scope_rt.SCOPERuntime,
        "_argv",
        lambda self, **kwargs: [str(tmp_path / "fake.py")],
    )

    runtime = scope_rt.SCOPERuntime(python_executable="/usr/bin/python3", default_work_dir=tmp_path / "scope")
    result = runtime.predict(
        images=image,
        action_path=action,
        output_path=out,
        prompt="hi",
        execute=True,
    )
    assert result["status"] == "success"
    assert calls
    assert calls[0]["timeout"] == 12.0
    assert Path(calls[0]["stdout_path"]).name.endswith(".scope.log")


def test_versecrafter_stage_uses_run_logged_subprocess(monkeypatch, tmp_path):
    from worldfoundry.synthesis.visual_generation.versecrafter import worldfoundry_runtime as vc

    calls: list[dict] = []

    def fake_logged(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vc, "run_logged_subprocess", fake_logged)
    monkeypatch.setattr(vc, "synthesis_timeout_seconds", lambda default=None: None)
    monkeypatch.setattr(vc, "ensure_in_tree_runtime", lambda *a, **k: tmp_path)

    runtime = vc.VerseCrafterRuntime(python_executable="/usr/bin/python3")
    runtime.repo_root = tmp_path
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    runtime._run_stage("depth", ["echo", "ok"], log_dir=log_dir, env={})
    assert calls
    assert calls[0]["stdout_path"] == log_dir / "depth.stdout.log"


def test_lingbot_video_run_plan_uses_run_logged_subprocess(monkeypatch, tmp_path):
    from worldfoundry.synthesis.visual_generation.lingbot_video import runtime as lb

    calls: list[dict] = []

    def fake_logged(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(lb, "run_logged_subprocess", fake_logged)
    monkeypatch.setattr(lb, "synthesis_timeout_seconds", lambda default=None: default)

    out = tmp_path / "out.mp4"
    out.write_bytes(b"mp4")
    plan = lb.LingBotVideoRuntimePlan(
        command=("python", "-c", "pass"),
        env={},
        workdir=str(tmp_path),
        checkpoint_dir=str(tmp_path),
        output_path=str(out),
        refiner_output_path=None,
        mode="t2v",
    )
    result = lb.LingBotVideoRuntime(checkpoint_dir=tmp_path).run_plan(
        plan, timeout_seconds=30, log_dir=tmp_path
    )
    assert calls
    assert calls[0]["timeout"] == 30.0
    assert Path(calls[0]["stdout_path"]).name == "lingbot_video_stdout.log"
    assert result["ok"] is True


def test_sy03_batch3_adapters_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "worldfoundry/synthesis/visual_generation/ac3d/runtime.py",
        "worldfoundry/synthesis/visual_generation/inspatio_world/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/forcing/runtime.py",
        "worldfoundry/synthesis/visual_generation/lingbot_video/runtime.py",
        "worldfoundry/synthesis/visual_generation/three_d_four_d/runtime.py",
        "worldfoundry/synthesis/visual_generation/world_model/runtime_manifest.py",
        "worldfoundry/synthesis/visual_generation/official_video_runtime.py",
        "worldfoundry/synthesis/visual_generation/kairos/runtime.py",
    )
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8")
        assert "run_logged_subprocess" in text, rel
        assert "synthesis_timeout_seconds" in text, rel


def test_sy03_batch4_adapters_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "worldfoundry/synthesis/visual_generation/magi/worldfoundry_runner.py",
        "worldfoundry/synthesis/visual_generation/pusa_vidgen/adapter.py",
        "worldfoundry/synthesis/visual_generation/unianimate_dit/worldfoundry_runner.py",
    )
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8")
        assert "run_logged_subprocess" in text, rel
        assert "synthesis_timeout_seconds" in text, rel
        assert "subprocess.run(" not in text, rel


def test_pusa_run_plan_uses_run_logged_subprocess(monkeypatch, tmp_path):
    from worldfoundry.synthesis.visual_generation.pusa_vidgen import adapter as pusa

    calls: list[dict] = []

    def fake_logged(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        out = Path(kwargs["stdout_path"]).parent
        (out / "clip.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pusa, "run_logged_subprocess", fake_logged)
    monkeypatch.setattr(pusa, "synthesis_timeout_seconds", lambda default=None: default)

    out = tmp_path / "out.mp4"
    plan = pusa.PusaVidGenRuntimePlan(
        command=("python", "-c", "pass"),
        env={},
        workdir=str(tmp_path),
        checkpoint_root=str(tmp_path),
        output_dir=str(tmp_path),
        output_path=str(out),
    )
    result = pusa.PusaVidGenRuntime(
        model_id="pusa",
        checkpoint_root=tmp_path,
        base_model_root=tmp_path,
    ).run_plan(plan, timeout_seconds=45, log_dir=tmp_path)
    assert calls
    assert calls[0]["timeout"] == 45.0
    assert Path(calls[0]["stdout_path"]).name == "pusa_vidgen_stdout.log"
    assert result["ok"] is True
