import importlib
import pkgutil
import csv
from pathlib import Path

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.base_models.three_dimensions.point_clouds.pi3 import SOURCE_ROOT as PI3_SOURCE_ROOT
from worldfoundry.synthesis.visual_generation.warp_as_history.variants import (
    WARP_AS_HISTORY_VARIANTS,
    get_warp_as_history_variant,
    runtime_root,
    test_cases_root as warp_as_history_test_cases_root,
)
import worldfoundry.pipelines.warp_as_history as wah_pipelines
from worldfoundry.pipelines.warp_as_history.pipeline_warp_as_history import WarpAsHistoryPipeline
from worldfoundry.studio.catalog import discover_catalog


def test_warp_as_history_runtime_uses_shared_base_models():
    root = runtime_root()
    assert (root / "scripts" / "infer_warp_as_history.py").is_file()
    assert (root / "warp_as_history" / "pipeline.py").is_file()
    assert (root / "helios" / "diffusers_version" / "pipeline_helios_diffusers.py").is_file()
    assert not (root / "third_party" / "Pi3").exists()
    assert not (root / "data" / "demo").exists()
    assert (PI3_SOURCE_ROOT / "pi3" / "models" / "pi3x.py").is_file()
    for variant in WARP_AS_HISTORY_VARIANTS.values():
        assert (warp_as_history_test_cases_root() / variant.demo_csv_path).is_file(), variant.model_id


def test_warp_as_history_runtime_logic_lives_under_synthesis():
    synthesis_root = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/synthesis/visual_generation/warp_as_history"
    )
    base_root = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/synthesis/visual_generation/warp_as_history"
    )
    synthesis_text = (synthesis_root / "warp_as_history_synthesis.py").read_text(encoding="utf-8")
    runtime_text = (base_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")

    assert (synthesis_root / "variants.py").is_file()
    assert (base_root / "variants.py").is_file()
    assert "class WarpAsHistoryRuntime" in runtime_text
    assert "subprocess.run" in runtime_text
    assert "def _write_csv" in runtime_text
    assert "worldfoundry.synthesis.visual_generation.warp_as_history" in synthesis_text
    assert "subprocess.run" not in synthesis_text
    assert "def _write_csv" not in synthesis_text
    assert "def _subprocess_env" not in synthesis_text


def test_all_warp_as_history_pipeline_modules_import():
    for module in pkgutil.iter_modules(wah_pipelines.__path__):
        if module.name.startswith("pipeline_"):
            importlib.import_module(f"{wah_pipelines.__name__}.{module.name}")


def test_warp_as_history_pipeline_plan_only_uses_official_runner(tmp_path):
    pipe = WarpAsHistoryPipeline.from_pretrained(lazy=True)

    result = pipe(
        prompt="a camera controlled validation test",
        output_path=tmp_path / "wah.mp4",
        plan_only=True,
        return_dict=True,
    )

    command = " ".join(result["command"])
    assert result["model_id"] == "warp-as-history"
    assert result["backend_quality"] == "execution_plan"
    assert result["artifact_kind"] == "generated_video"
    assert result["artifact_path"].endswith("wah.json")
    assert "warp_as_history_runtime/scripts/infer_warp_as_history.py" in command
    assert "checkpoints/helios-distilled" in command
    assert "visible_lora_state_step1000.safetensors" in command
    with Path(result["command"][2]).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert Path(row["first_frame_path"]).is_file()
    assert Path(row["warp_video_path"]).is_file()
    assert Path(row["warp_visibility_mask_path"]).is_file()


def test_warp_as_history_catalog_entries_include_aliases():
    entry = load_model_zoo_registry().get("warp-as-history")

    assert entry.pipeline_target == "worldfoundry.pipelines.warp_as_history.pipeline_warp_as_history:WarpAsHistoryPipeline"
    assert {"wah", "warp_as_history", "yyfz233/warp-as-history"}.issubset(set(entry.aliases))
    assert get_warp_as_history_variant("wah").model_id == "warp-as-history"


def test_warp_as_history_is_discoverable_by_studio_catalog():
    ids = {entry.model_id for entry in discover_catalog()}
    assert "warp-as-history" in ids
