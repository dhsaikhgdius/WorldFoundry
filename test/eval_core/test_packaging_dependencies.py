from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from tools.packaging import build_sanitized_sdist as sdist_builder
from tools.packaging import check_release_worktree
from tools.packaging.check_sdist_hygiene import check_sdist_hygiene

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

    assert (REPO_ROOT / "worldfoundry/studio/assets/openenvision-logo.png").is_file()
    assert "assets/**/*" in package_data["worldfoundry.studio"]


def test_unified_requirements_install_data_gpu_probe_dependencies() -> None:
    requirements = (REPO_ROOT / "requirements" / "worldfoundry-unified.txt").read_text(encoding="utf-8")
    install_script = (REPO_ROOT / "scripts" / "setup" / "conda_install.sh").read_text(encoding="utf-8")

    assert "build" in requirements
    assert "-e .[tui,optimized_core,video,hf,api,ui,metrics]" in requirements
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


def test_development_and_ci_install_sdist_hygiene_dependencies() -> None:
    optional = _optional_dependencies()
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    dev_tools = (REPO_ROOT / "scripts" / "dev" / "check_dev_tools.py").read_text(encoding="utf-8")

    assert "build" in optional["dev"]
    assert "$(PIP) install -e ." in makefile
    assert "$(PIP) install --no-deps -e ." not in makefile
    assert "$(PIP) install build pre-commit pytest PyYAML ruff" in makefile
    assert "sdist-hygiene:" in makefile
    assert "python -m pip install build pre-commit pytest PyYAML ruff" in workflow
    assert "run: make sdist-hygiene" in workflow
    assert 'REPO_ROOT / "tools" / "packaging"' in dev_tools
    assert "make sdist-hygiene" in dev_tools


def test_sdist_manifest_excludes_generated_downloaded_and_large_artifacts() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    required_snippets = (
        "include worldfoundry/data/benchmarks/*.md",
        "recursive-include worldfoundry/data/benchmarks/catalog *.yaml",
        "recursive-include worldfoundry/data/benchmarks/runtime_profiles *.yaml",
        "recursive-include worldfoundry/data/models/runtime_profiles *.yaml",
        "recursive-include worldfoundry/data/models/runtime *.yaml",
        "recursive-include worldfoundry/data/models/bindings *.yaml",
        "recursive-include worldfoundry/data/models/runtime/configs *.yaml",
        "recursive-include worldfoundry/data/models/runtime/configs *.yml",
        "recursive-include worldfoundry/data/models/runtime/configs *.json",
        "prune tmp",
        "prune cache",
        "prune data/hfd_datasets",
        "prune worldfoundry/data/test_cases",
        "prune docs/fumadocs/.next",
        "prune docs/fumadocs/node_modules",
        "prune worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/tokenizer/notebook",
        "prune worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/tokenizer/test_data",
        "prune worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2/cosmos_predict2/_src/imaginaire/attention/tests",
        "prune worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2/cosmos_predict2/_src/predict2/tests",
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


def test_sanitized_sdist_builder_uses_tracked_release_sources_plus_release_control_files() -> None:
    builder = (REPO_ROOT / "tools" / "packaging" / "build_sanitized_sdist.py").read_text(encoding="utf-8")

    assert '"git", "ls-files", "-z", "--"' in builder
    assert '"--others"' not in builder
    assert "UNTRACKED_SOURCE_INCLUDE_PATHS" not in builder
    assert "EXTRA_WORKTREE_FILES" in builder
    assert '"OPEN_SOURCE_CHECKLIST.md"' in builder
    assert '"worldfoundry/data/test_cases"' in builder
    assert "_commit_sanitized_tree(tree)" in builder
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WORLDFOUNDRY" in builder


def test_sanitized_sdist_builder_sets_stable_version_environment() -> None:
    env = sdist_builder._build_environment("0.1.dev-test")

    assert env["SETUPTOOLS_SCM_PRETEND_VERSION"] == "0.1.dev-test"
    assert env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WORLDFOUNDRY"] == "0.1.dev-test"
    assert env["PIP_INDEX_URL"].startswith("https://")
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"


def test_release_worktree_status_parser_reports_dirty_entries() -> None:
    issues = check_release_worktree.parse_status_lines(
        [
            " M pyproject.toml",
            "D  obsolete.py",
            "?? worldfoundry/new_runtime.py",
        ]
    )

    assert [issue.to_json() for issue in issues] == [
        {"status": " M", "path": "pyproject.toml"},
        {"status": "D ", "path": "obsolete.py"},
        {"status": "??", "path": "worldfoundry/new_runtime.py"},
    ]


def test_release_worktree_report_json_includes_summary_counts() -> None:
    issues = check_release_worktree.parse_status_lines(
        [
            " M pyproject.toml",
            "?? worldfoundry/new_runtime.py",
            "?? docs/new.md",
        ]
    )
    report = check_release_worktree.WorktreeReport(
        issue_count=len(issues),
        issues=issues,
        status_counts={"??": 2, " M": 1},
        top_path_counts={"src": 1, "docs": 1, "pyproject.toml": 1},
        untracked_directory_counts={
            "worldfoundry/new_runtime.py": 1,
            "docs/new.md": 1,
        },
        untracked_extension_counts={".py": 1, ".md": 1},
    )

    assert report.to_json()["status_counts"] == {"??": 2, " M": 1}
    assert report.to_json()["top_path_counts"] == {"src": 1, "docs": 1, "pyproject.toml": 1}
    assert report.to_json()["untracked_directory_counts"] == {
        "worldfoundry/new_runtime.py": 1,
        "docs/new.md": 1,
    }
    assert report.to_json()["untracked_extension_counts"] == {".py": 1, ".md": 1}


def test_release_worktree_issue_summary_groups_untracked_source_trees() -> None:
    issues = check_release_worktree.parse_status_lines(
        [
            " M pyproject.toml",
            "?? worldfoundry/synthesis/visual_generation/demo/runtime/a.py",
            "?? worldfoundry/synthesis/visual_generation/demo/runtime/b.hpp",
            "?? scripts/inference/run_demo",
        ]
    )

    summary = check_release_worktree.summarize_issues(issues)

    assert summary["status_counts"] == {"??": 3, " M": 1}
    assert summary["top_path_counts"] == {"src": 2, "scripts": 1, "pyproject.toml": 1}
    assert summary["untracked_directory_counts"] == {
        "worldfoundry/synthesis/visual_generation/demo/runtime": 2,
        "scripts/inference/run_demo": 1,
    }
    assert summary["untracked_extension_counts"] == {".hpp": 1, ".py": 1, "[no-ext]": 1}


def _write_tarball(tarball: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(tarball, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_sdist_hygiene_checker_accepts_clean_source_archives(tmp_path: Path) -> None:
    tarball = tmp_path / "clean.tar.gz"
    _write_tarball(
        tarball,
        {
            "worldfoundry-0.1/MANIFEST.in": b"include worldfoundry/data/benchmarks/catalog *.yaml\n",
            "worldfoundry-0.1/.pre-commit-config.yaml": b"repos: []\n",
            "worldfoundry-0.1/README.md": b"# WorldFoundry\n",
            "worldfoundry-0.1/worldfoundry/cli/main.py": b"",
            "worldfoundry-0.1/worldfoundry/evaluation/framework.py": b"",
            "worldfoundry-0.1/worldfoundry/evaluation/__init__.py": b"",
            "worldfoundry-0.1/worldfoundry/runtime/env.py": b"",
            "worldfoundry-0.1/worldfoundry/data/benchmarks/catalog/video_world_benchmarks.yaml": b"[]",
            "worldfoundry-0.1/tools/packaging/build_sanitized_sdist.py": b"",
            "worldfoundry-0.1/tools/packaging/check_sdist_hygiene.py": b"",
            "worldfoundry-0.1/tools/packaging/check_release_worktree.py": b"",
        },
    )

    report = check_sdist_hygiene(tarball, max_size_mib=1.0)

    assert report.ok
    assert report.file_count == 11
    assert report.issues == ()


def test_sdist_hygiene_checker_rejects_upstream_runtime_test_dirs(tmp_path: Path) -> None:
    tarball = tmp_path / "cosmos-tests.tar.gz"
    _write_tarball(
        tarball,
        {
            "worldfoundry-0.1/MANIFEST.in": b"",
            "worldfoundry-0.1/worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2/cosmos_predict2/_src/imaginaire/attention/tests/sdpa_test.py": b"",
            "worldfoundry-0.1/worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2/cosmos_predict2/_src/predict2/tests/test_model.py": b"",
            "worldfoundry-0.1/worldfoundry/cli/main.py": b"",
            "worldfoundry-0.1/worldfoundry/evaluation/framework.py": b"",
            "worldfoundry-0.1/worldfoundry/runtime/env.py": b"",
            "worldfoundry-0.1/worldfoundry/data/benchmarks/catalog/video_world_benchmarks.yaml": b"[]",
            "worldfoundry-0.1/tools/packaging/build_sanitized_sdist.py": b"",
            "worldfoundry-0.1/tools/packaging/check_sdist_hygiene.py": b"",
        },
    )

    report = check_sdist_hygiene(tarball, max_size_mib=1.0)

    forbidden = {issue.path for issue in report.issues if "forbidden path segment" in issue.reason}
    assert not report.ok
    assert any("imaginaire/attention/tests" in path for path in forbidden)
    assert any("_src/predict2/tests" in path for path in forbidden)


def test_sdist_hygiene_checker_rejects_missing_release_critical_members(tmp_path: Path) -> None:
    tarball = tmp_path / "missing-critical.tar.gz"
    _write_tarball(tarball, {"worldfoundry-0.1/README.md": b"# WorldFoundry\n"})

    report = check_sdist_hygiene(tarball, max_size_mib=1.0)
    missing_paths = {issue.path for issue in report.issues if issue.reason == "required source member missing from sdist"}

    assert not report.ok
    assert "worldfoundry/evaluation/framework.py" in missing_paths
    assert "tools/packaging/build_sanitized_sdist.py" in missing_paths


def test_sdist_hygiene_checker_rejects_release_artifacts_and_unsafe_paths(tmp_path: Path) -> None:
    tarball = tmp_path / "dirty.tar.gz"
    _write_tarball(
        tarball,
        {
            "worldfoundry-0.1/data/hfd_datasets/sample.json": b"{}",
            "worldfoundry-0.1/worldfoundry/synthesis/model.safetensors": b"weights",
            "worldfoundry-0.1/docs/fumadocs/.next/server/app.js": b"bundle",
            "../escape.txt": b"bad",
        },
    )

    report = check_sdist_hygiene(tarball, max_size_mib=1.0)
    reasons = {issue.reason for issue in report.issues}

    assert not report.ok
    assert any("data/hfd_datasets" in issue.path for issue in report.issues)
    assert any(issue.path.endswith("model.safetensors") for issue in report.issues)
    assert any("docs/fumadocs/.next" in issue.path for issue in report.issues)
    assert "tar member path escapes the archive root" in reasons


def test_sdist_hygiene_checker_enforces_size_limit(tmp_path: Path) -> None:
    tarball = tmp_path / "oversized.tar.gz"
    _write_tarball(tarball, {"worldfoundry-0.1/README.md": b"# WorldFoundry\n"})

    report = check_sdist_hygiene(tarball, max_size_mib=0.0)

    assert not report.ok
    assert report.issues[0].path == str(tarball)
    assert "compressed sdist" in report.issues[0].reason
