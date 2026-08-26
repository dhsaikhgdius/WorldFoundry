"""D-07: Dockerfile exposes base-runtime / base-devel / cpu multi-stage targets."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
BUILD = REPO_ROOT / "docker" / "build_with_docker.sh"


def test_d07_dockerfile_defines_runtime_devel_cpu_stages() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "AS base-runtime" in text
    assert "AS base-devel" in text
    assert "AS cpu" in text
    assert "FROM python:3.11-slim AS cpu" in text
    assert "cudnn-runtime" in text
    # Default final export remains devel for backward compatibility.
    assert text.rstrip().endswith("FROM base-devel") or "FROM base-devel\n" in text.split("# Default export")[-1]


def test_d07_build_script_supports_target_flag() -> None:
    text = BUILD.read_text(encoding="utf-8")
    assert "--target" in text
    assert 'TARGET="${WORLDFOUNDRY_DOCKER_TARGET:-base-devel}"' in text
    assert '--target "${TARGET}"' in text
    assert "CUDA_RUNTIME_IMAGE" in text
    # Preserve D-03 defaults.
    assert 'PLATFORMS="linux/amd64"' in text
    assert "WORLDFOUNDRY_DOCKER_BUILD_HOST_NETWORK" in text
