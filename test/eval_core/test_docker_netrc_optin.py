"""Docker test runner mounts ~/.netrc only when WORLDFOUNDRY_DOCKER_MOUNT_NETRC=1."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "test" / "run_tests_docker.sh"


def test_run_tests_docker_netrc_mount_is_opt_in() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_DOCKER_MOUNT_NETRC" in text
    assert 'WORLDFOUNDRY_DOCKER_MOUNT_NETRC:-0' in text
    # Unconditional existence check must not be the sole gate.
    assert "if [[ -f \"${HOME}/.netrc\" ]]; then" not in text
