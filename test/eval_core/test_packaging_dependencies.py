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

    # Model runtime profiles moved from data/models/runtime_profiles to
    # data/models/runtime (+ environments/), and the cosmos vendored trees are
    # covered by the global binary/notebook excludes instead of per-path prunes.
    required_snippets = (
        "include worldfoundry/data/benchmarks/*.md",
        "recursive-include worldfoundry/data/benchmarks/catalog *.yaml",
        "recursive-include worldfoundry/data/benchmarks/runtime_profiles *.yaml",
        "recursive-include worldfoundry/data/models/catalog *.yaml",
        "recursive-include worldfoundry/data/models/runtime *.yaml",
        "recursive-include worldfoundry/data/models/bindings *.yaml",
        "recursive-include worldfoundry/data/models/runtime/configs *.yaml",
        "recursive-include worldfoundry/data/models/runtime/configs *.yml",
        "recursive-include worldfoundry/data/models/runtime/configs *.json",
        "recursive-include worldfoundry/data/models/runtime/environments *.yaml",
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


def test_eval_core_extra_declares_cpu_release_gate_dependencies() -> None:
    """EX-04: contributors can install ``.[eval_core]`` for make test-eval-core."""

    optional = _optional_dependencies()
    assert "eval_core" in optional
    joined = "\n".join(optional["eval_core"])
    assert "worldfoundry[test,ui]" in joined or "worldfoundry[test]" in joined
    assert any(item == "torch" or item.startswith("torch") for item in optional["eval_core"])
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert '.[eval_core]' in makefile or ".[eval_core]" in makefile
