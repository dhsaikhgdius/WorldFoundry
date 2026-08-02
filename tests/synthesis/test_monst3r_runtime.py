from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.utils import REPO_ROOT
from worldfoundry.synthesis.visual_generation.three_d_four_d.runtime import (
    ThreeDFourDRuntimeSynthesis,
    three_d_four_d_runtime_spec,
)


MONST3R_ROOT = REPO_ROOT / "worldfoundry" / "base_models" / "three_dimensions" / "general_3d" / "monst3r"
KITCHEN_IMAGES = REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "vggt" / "examples" / "kitchen" / "images"


def test_monst3r_runtime_spec_uses_in_tree_demo() -> None:
    spec = three_d_four_d_runtime_spec("monst3r")

    assert spec.entrypoint == "demo.py"
    assert spec.command_kind == "monst3r_demo"
    assert spec.artifact_filename == "monst3r.glb"
    assert (MONST3R_ROOT / spec.entrypoint).is_file()


def test_monst3r_plan_uses_in_tree_source_and_staged_multi_image_dir(tmp_path: Path) -> None:
    spec = three_d_four_d_runtime_spec("monst3r")
    runtime = ThreeDFourDRuntimeSynthesis(
        spec=spec,
        source_root=MONST3R_ROOT,
        device="cpu",
        options={
            "weights": "/tmp/monst3r.pth",
            "flow_loss_weight": 0.0,
            "skip_pair_dynamic_mask": True,
            "silent": True,
        },
    )

    result = runtime.predict(
        images=[KITCHEN_IMAGES / "00.png", KITCHEN_IMAGES / "01.png"],
        output_path=tmp_path / "scene.glb",
        run_dir=tmp_path / "run",
        plan_only=True,
    )

    command = result["command"]
    input_dir = Path(command[command.index("--input_dir") + 1])
    assert result["status"] == "prepared"
    assert command[1] == str((MONST3R_ROOT / "demo.py").resolve())
    assert "--flow_loss_weight" in command
    assert command[command.index("--flow_loss_weight") + 1] == "0.0"
    assert "--skip_pair_dynamic_mask" in command
    assert input_dir == tmp_path / "run" / "monst3r_inputs"
    assert sorted(path.name for path in input_dir.iterdir()) == ["00000.png", "00001.png"]


def test_three_d_four_d_runtime_reuses_shared_torch_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    monkeypatch.delenv("TORCH_HOME", raising=False)
    runtime = ThreeDFourDRuntimeSynthesis(
        spec=three_d_four_d_runtime_spec("monst3r"),
        source_root=MONST3R_ROOT,
        device="cpu",
    )

    assert runtime._subprocess_env()["TORCH_HOME"] == str(tmp_path / "torch_hub")
