from __future__ import annotations

from pathlib import Path

import pytest

# NOTE: the sanitized-sdist builder tests that used to live in this module were
# removed together with the tools/packaging toolchain (build_sanitized_sdist,
# check_release_worktree, check_sdist_hygiene no longer exist in the repo).
# Only the packaging-metadata contract tests below remain.

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 only.
    tomllib = pytest.importorskip("tomli")


REPO_ROOT = Path(__file__).resolve().parents[2]


def _optional_dependencies() -> dict[str, list[str]]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    return payload["project"]["optional-dependencies"]


def _package_data() -> dict[str, list[str]]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    return payload["tool"]["setuptools"]["package-data"]


def test_data_gpu_probe_dependencies_are_packaged() -> None:
    optional = _optional_dependencies()

    for extra in ("video", "metrics", "all"):
        assert "pyarrow" in optional[extra]
        assert "h5py" in optional[extra]


def test_studio_static_assets_are_packaged() -> None:
    package_data = _package_data()

    # The OpenEnvision logo moved to the CLI package and the PNG itself is
    # gitignored (worldfoundry/cli/assets/*.png), so the checked-in contract is
    # the packaging metadata plus the loader's asset path, not the binary file.
    assert "assets/openenvision_logo.png" in package_data["worldfoundry.cli"]
    assert "assets/**/*" in package_data["worldfoundry.studio"]
    assert (REPO_ROOT / "worldfoundry/studio/assets").is_dir()

    from worldfoundry.cli.tui_brand import logo_asset_path

    assert logo_asset_path().name == "openenvision_logo.png"


def test_unified_requirements_install_data_gpu_probe_dependencies() -> None:
    requirements = (REPO_ROOT / "requirements" / "worldfoundry-unified.txt").read_text(encoding="utf-8")
    install_script = (REPO_ROOT / "scripts" / "setup" / "conda_install.sh").read_text(encoding="utf-8")

    assert "build" in requirements
    assert (
        "-e .[tui,optimized_core,video,hf,api,ui,metrics,studio_pointcloud,studio_rerun]"
        in requirements
    )
    for package in ("h5py", "numpy", "opencv-python", "pillow", "pyarrow"):
        assert package in requirements
    for module in ('"cv2"', '"h5py"', '"numpy"', '"PIL"', '"pyarrow"'):
        assert module in install_script


def test_optimized_core_extra_declares_flashdreams_runtime_dependencies() -> None:
    optional = _optional_dependencies()

    assert "optimized_core" in optional
    assert "boto3" in optional["optimized_core"]
    assert "einops>=0.8.0,<0.9.0" in optional["optimized_core"]
    assert "filelock>=3.24.2,<4.0.0" in optional["optimized_core"]
    assert "huggingface-hub" in optional["optimized_core"]
    assert "loguru" in optional["optimized_core"]
    assert "safetensors>=0.4" in optional["optimized_core"]
    assert "torch>=2.7,<2.12.0" in optional["optimized_core"]


def test_sdist_manifest_excludes_generated_downloaded_and_large_artifacts() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    # Model runtime profiles live under data/models/runtime (+ configs/environments/).
    # DA-06: one recursive-include on runtime covers nested configs/environments;
    # eval_configs must be listed explicitly (not only via package-data globs).
    required_snippets = (
        "include worldfoundry/data/benchmarks/*.md",
        "recursive-include worldfoundry/data/benchmarks/catalog *.yaml",
        "recursive-include worldfoundry/data/benchmarks/runtime_profiles *.yaml",
        "recursive-include worldfoundry/data/benchmarks/eval_configs *.yaml *.yml",
        "recursive-include worldfoundry/data/models/catalog *.yaml",
        "recursive-include worldfoundry/data/models/bindings *.yaml",
        "recursive-include worldfoundry/data/models/runtime *.yaml *.yml *.json",
        "prune tmp",
        "prune cache",
        "prune data/hfd_datasets",
        "prune worldfoundry/data/test_cases",
        "prune docs/fumadocs/.next",
        "prune docs/fumadocs/node_modules",
        "prune worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/ChatUniVi/eval",
        "prune worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/ChatUniVi/train",
        "prune worldfoundry/synthesis/visual_generation/dynamicrafter_pandora/DynamiCrafter/assets",
        "prune worldfoundry/synthesis/visual_generation/dynamicrafter_pandora/DynamiCrafter/prompts",
        "global-exclude *.py[cod]",
    )
    for snippet in required_snippets:
        assert snippet in manifest
    # Nested runtime paths must not be re-listed (SSOT: one runtime include).
    for redundant in (
        "recursive-include worldfoundry/data/models/runtime/configs",
        "recursive-include worldfoundry/data/models/runtime/environments",
        "recursive-include worldfoundry/data/models/runtime *.yaml\n",
        "recursive-include worldfoundry/data/models/runtime *.json\n",
    ):
        assert redundant not in manifest
    for pattern in (
        "*.gif",
        "*.ipynb",
        "*.pdf",
        "*.mp4",
        "*.safetensors",
        "*.npy",
        "*.pt",
        "*.pth",
        "*.onnx",
    ):
        assert pattern in manifest


def test_package_data_ssot_covers_benchmark_eval_configs() -> None:
    """Wheel SSOT: broad benchmarks/** globs must ship eval_configs YAML."""
    package_data = _package_data()
    data_globs = package_data["worldfoundry.data"]
    assert "benchmarks/**/*.yaml" in data_globs
    assert "benchmarks/**/*.yml" in data_globs
    assert "models/**/*.yaml" in data_globs

    eval_configs_root = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "eval_configs"
    assert eval_configs_root.is_dir()
    sample = eval_configs_root / "embodied" / "libero" / "spatial.yaml"
    assert sample.is_file()
    # Positive sdist contract: MANIFEST lists eval_configs (not package-data alone).
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include worldfoundry/data/benchmarks/eval_configs" in manifest
