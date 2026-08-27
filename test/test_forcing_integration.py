from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from worldfoundry.evaluation.models import discover_model_registry
from worldfoundry.operators.forcing_operator import CausalForcingOperator, SelfForcingOperator
from worldfoundry.pipelines.forcing.pipeline_forcing import CausalForcingPipeline
from worldfoundry.synthesis.visual_generation.forcing import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.forcing.runtime import SelfForcingRuntime


def test_self_forcing_operator_accepts_prompt_and_optional_image() -> None:
    operator = SelfForcingOperator()
    operator.get_interaction(123)

    interaction = operator.process_interaction()
    perception = operator.process_perception(images="/tmp/input.png")
    prompt = operator.process_prompt("A detailed cinematic scene.")

    assert interaction["seed"] == 123
    assert perception["images"] == "/tmp/input.png"
    assert prompt["prompt"] == "A detailed cinematic scene."


def test_causal_forcing_operator_accepts_prompt_and_optional_image() -> None:
    operator = CausalForcingOperator()
    operator.get_interaction(7)

    interaction = operator.process_interaction()
    perception = operator.process_perception(images="/tmp/input.png")
    prompt = operator.process_prompt("A detailed causal scene.")

    assert interaction["seed"] == 7
    assert perception["images"] == "/tmp/input.png"
    assert prompt["prompt"] == "A detailed causal scene."


def test_causal_forcing_pipeline_forwards_model_specific_runtime_options() -> None:
    pipe = CausalForcingPipeline.from_pretrained(
        model_path={
            "runtime_root": "/tmp/Causal-Forcing",
            "checkpoint_path": "/tmp/causal.pt",
            "config_path": "/tmp/config.yaml",
            "wan_models_root": "/tmp/wan",
        },
        model_id="causal-forcing",
        device="cuda:2",
    )

    runtime = pipe.synthesis_model.runtime
    assert runtime.model_id == "causal-forcing"
    assert runtime.runtime_root == "/tmp/Causal-Forcing"
    assert runtime.checkpoint_path == "/tmp/causal.pt"
    assert runtime.config_path == "/tmp/config.yaml"
    assert runtime.wan_models_root == "/tmp/wan"
    assert runtime.device == "cuda:2"


def test_forcing_runtime_builds_official_command(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "Self-Forcing"
    wan_root = tmp_path / "ckpt"
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    config = runtime_root / "configs" / "self_forcing_dmd.yaml"
    (runtime_root / "configs").mkdir(parents=True)
    (wan_root / "Wan2.1-T2V-1.3B").mkdir(parents=True)
    (wan_root / "Wan2.1-T2V-14B").mkdir(parents=True)
    (runtime_root / "inference.py").write_text("print('stub')\n", encoding="utf-8")
    config.write_text("generator_ckpt: checkpoints/ode_init.pt\n", encoding="utf-8")
    checkpoint.write_bytes(b"stub")

    captured = {}

    def fake_run(command, *, stdout_path, stderr_path, cwd, env, start_new_session):
        del stdout_path, stderr_path, env, start_new_session
        captured["command"] = command
        captured["cwd"] = cwd
        output_dir = Path(command[command.index("--output_folder") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "0-0_ema.mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(runtime_module, "run_logged_subprocess", fake_run)
    monkeypatch.setattr(
        runtime_module,
        "_load_video_frames",
        lambda _: np.zeros((2, 8, 8, 3), dtype=np.uint8),
    )

    model = SelfForcingRuntime(
        runtime_root=runtime_root,
        checkpoint_path=checkpoint,
        config_path=config,
        wan_models_root=wan_root,
        python_executable="/tmp/python",
    )
    result = model.predict(
        prompt="demo prompt",
        output_path=tmp_path / "result.mp4",
        num_output_frames=2,
        show_progress=False,
    )

    command = captured["command"]
    assert command[:2] == ["/tmp/python", str((runtime_root / "inference.py").resolve())]
    assert command[command.index("--config_path") + 1] == str(config.resolve())
    assert command[command.index("--checkpoint_path") + 1] == str(checkpoint.resolve())
    assert command[command.index("--num_output_frames") + 1] == "2"
    assert "--save_with_index" in command
    assert "--use_ema" in command
    assert Path(captured["cwd"]).name == "runtime_cwd"
    assert result["artifact_path"] == str((tmp_path / "result.mp4").resolve())
    assert result["video"].shape == (2, 8, 8, 3)


def test_model_registry_contains_forcing_models() -> None:
    registry = discover_model_registry()
    self_model = registry.get("self-forcing")
    causal_model = registry.get("causal-forcing")

    assert self_model.has_loader is True
    assert self_model.has_infer is True
    assert self_model.pipeline_target == "worldfoundry.pipelines.forcing.pipeline_forcing:SelfForcingPipeline"
    assert causal_model.has_loader is True
    assert causal_model.has_infer is True
    assert causal_model.pipeline_target == "worldfoundry.pipelines.forcing.pipeline_forcing:CausalForcingPipeline"
