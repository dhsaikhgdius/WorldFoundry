"""D-04: test/run_tests_docker.sh defaults to the published worldfoundry base image."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "test" / "run_tests_docker.sh"


def test_d04_run_tests_docker_defaults_to_worldfoundry_base() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'WORLDFOUNDRY_TEST_IMAGE:-ghcr.io/openenvision/worldfoundry:base' in text
    assert "Default: ghcr.io/openenvision/worldfoundry:base" in text
    # Bare nvidia/cuda must not remain the default assignment.
    assert "WORLDFOUNDRY_TEST_IMAGE:-nvidia/cuda:" not in text
