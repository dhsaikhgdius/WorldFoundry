from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from worldfoundry.evaluation.models import discover_model_registry
from worldfoundry.operators.kairos_operator import KairosOperator
from worldfoundry.pipelines.kairos.pipeline_kairos import KairosPipeline
from worldfoundry.synthesis.visual_generation.kairos import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.kairos.runtime import KairosRuntime


def test_kairos_operator_accepts_prompt_and_image() -> None:
    operator = KairosOperator()
    operator.get_interaction(9)

    assert operator.process_interaction()["seed"] == 9
    assert operator.process_prompt("A waterfall.")["prompt"] == "A waterfall."
    assert operator.process_perception(images="/tmp/input.png")["images"] == "/tmp/input.png"


def test_kairos_pipeline_forwards_runtime_options() -> None:
    pipe = KairosPipeline.from_pretrained(
        model_path={
            "runtime_root": "/tmp/kairos",
            "models_root": "/tmp/models",
            "config_path": "/tmp/config.py",
            "variant": "720p",
        },
        device="cuda:1",
    )

    runtime = pipe.synthesis_model.runtime
    assert runtime.runtime_root == "/tmp/kairos"
    assert runtime.models_root == "/tmp/models"
    assert runtime.config_path == "/tmp/config.py"
    assert runtime.variant == "720p"
    assert runtime.device == "cuda:1"


def test_kairos_runtime_builds_torchrun_command(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "kairos-sensenova"
    models_root = tmp_path / "models"
    config = runtime_root / "kairos" / "configs" / "kairos_4b_config_DMD.py"
    (runtime_root / "examples").mkdir(parents=True)
    (runtime_root / "kairos" / "third_party").mkdir(parents=True)
    config.parent.mkdir(parents=True)
    (runtime_root / "examples" / "inference.py").write_text("print('stub')\n", encoding="utf-8")
    (runtime_root / "kairos" / "third_party" / "manage_libs.py").write_text("print('libs')\n", encoding="utf-8")
    config.write_text(
        "KAIROS_MODEL_DIR = 'models'\n"
        "pipeline = {'pipeline_args': {'vae_path': 'old', 'text_encoder_path': 'old'}}\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(command, check, cwd, env, stdout=None, stderr=None, text=None):
        del check, env, stdout, stderr, text
        calls.append((command, cwd))
        if "torchrun" in command[0]:
            output_dir = Path(command[command.index("--input_file") + 1]).read_text(encoding="utf-8")
            marker = '"output_dir": "'
            target = output_dir.split(marker, 1)[1].split('"', 1)[0]
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "output.mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runtime_module,
        "_load_video_frames",
        lambda _: np.zeros((2, 8, 8, 3), dtype=np.uint8),
    )

    model = KairosRuntime(
        runtime_root=runtime_root,
        models_root=models_root,
        config_path=config,
        python_executable="/tmp/python",
        torchrun_executable="torchrun",
        defaults={"run_manage_libs": True},
    )
    result = model.predict(
        prompt="demo",
        output_path=tmp_path / "result.mp4",
        num_frames=2,
        run_manage_libs=True,
    )

    assert calls[0][0][:2] == ["/tmp/python", str(runtime_root / "kairos" / "third_party" / "manage_libs.py")]
    command = calls[1][0]
    assert command[0] == "torchrun"
    assert command[command.index("--nproc-per-node") + 1] == "1"
    assert command[command.index("--config_file") + 1].endswith("kairos_config.py")
    assert result["artifact_path"] == str((tmp_path / "result.mp4").resolve())
    assert result["video"].shape == (2, 8, 8, 3)


def test_model_registry_contains_kairos() -> None:
    registry = discover_model_registry()
    model = registry.get("kairos-sensenova")

    assert model.has_loader is True
    assert model.has_infer is True
    assert model.pipeline_target == "worldfoundry.pipelines.kairos.pipeline_kairos:KairosPipeline"
