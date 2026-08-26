"""D-05: docker/Dockerfile pins uv and aligns default Python to 3.11."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"


def test_d05_dockerfile_pins_uv_version() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG UV_VERSION=0.12.6" in text
    assert "ghcr.io/astral-sh/uv:${UV_VERSION}" in text
    assert "ghcr.io/astral-sh/uv:latest" not in text


def test_d05_dockerfile_aligns_python_311() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG PYTHON_VERSION=3.11" in text
    assert 'uv python install "${PYTHON_VERSION}"' in text
    assert "UV_PYTHON=${PYTHON_VERSION}" in text
