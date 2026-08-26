"""DA-08: closed model runtime task_family vocab + placement rules."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.models.runtime.profiles import (
    ALLOWED_MODEL_RUNTIME_TASK_FAMILIES,
    RuntimeProfile,
    load_runtime_profile_manifests,
)
from worldfoundry.evaluation.utils import (
    BENCHMARK_RUNTIME_PROFILE_DIR,
    MODEL_RUNTIME_CONFIGS_ROOT,
    MODEL_RUNTIME_PROFILES_ROOT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_allowed_task_family_vocab_is_closed_and_excludes_benchmark() -> None:
    assert "benchmark" not in ALLOWED_MODEL_RUNTIME_TASK_FAMILIES
    assert "video_generation" in ALLOWED_MODEL_RUNTIME_TASK_FAMILIES
    assert "image_generation" in ALLOWED_MODEL_RUNTIME_TASK_FAMILIES
    assert "vla_policy" in ALLOWED_MODEL_RUNTIME_TASK_FAMILIES


def test_bundled_model_runtime_profiles_use_closed_task_family_vocab() -> None:
    profiles = load_runtime_profile_manifests(MODEL_RUNTIME_PROFILES_ROOT)
    assert profiles
    used = {profile.task_family for profile in profiles}
    assert used <= ALLOWED_MODEL_RUNTIME_TASK_FAMILIES
    # On-disk YAML must declare task_family (no silent default-only coverage).
    for path in sorted(MODEL_RUNTIME_PROFILES_ROOT.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert payload.get("task_family"), f"{path.name} missing explicit task_family"


def test_runtime_profile_rejects_benchmark_task_family() -> None:
    with pytest.raises(ValueError, match="unsupported task_family"):
        RuntimeProfile.from_mapping(
            {
                "model_id": "fake-bench",
                "display_name": "Fake Bench",
                "task_family": "benchmark",
                "artifact_kind": "benchmark_scorecard",
                "artifact_filename": "scorecard.json",
            }
        )


def test_benchmark_profiles_must_not_live_under_model_runtime_profiles() -> None:
    model_ids = {path.stem for path in MODEL_RUNTIME_PROFILES_ROOT.glob("*.yaml")}
    benchmark_ids = {
        path.stem
        for path in (BENCHMARK_RUNTIME_PROFILE_DIR / "official").glob("*.yaml")
        if path.is_file()
    }
    # 4dworldbench is the known misplaced case; keep model ids disjoint from
    # official benchmark runtime profile ids when task_family would be benchmark.
    overlap = sorted(model_ids & benchmark_ids)
    for model_id in overlap:
        payload = yaml.safe_load((MODEL_RUNTIME_PROFILES_ROOT / f"{model_id}.yaml").read_text(encoding="utf-8"))
        assert payload.get("task_family") != "benchmark", model_id
    assert "4dworldbench" not in model_ids
    assert (BENCHMARK_RUNTIME_PROFILE_DIR / "official" / "4dworldbench.yaml").is_file()


def test_model_runtime_configs_forbid_same_stem_file_and_directory() -> None:
    root = MODEL_RUNTIME_CONFIGS_ROOT
    dirs = {path.name for path in root.iterdir() if path.is_dir()}
    yaml_stems = {path.stem for path in root.glob("*.yaml") if path.is_file()}
    yml_stems = {path.stem for path in root.glob("*.yml") if path.is_file()}
    overlap = sorted(dirs & (yaml_stems | yml_stems))
    assert not overlap, f"configs stem collision between file and directory: {overlap}"
    # infinite_world lives only as a directory after DA-08.
    assert (root / "infinite_world").is_dir()
    assert not (root / "infinite_world.yaml").exists()
    assert (root / "infinite_world" / "model.yaml").is_file()
