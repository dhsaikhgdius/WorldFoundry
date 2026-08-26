"""D-03: docker/build_with_docker.sh defaults to amd64; host network is opt-in."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "build_with_docker.sh"


def test_d03_push_default_is_linux_amd64_only() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'PLATFORMS="linux/amd64"' in text
    assert "linux/amd64,linux/arm64" not in text.split("if [[ -z \"${PLATFORMS}\" ]]; then", 1)[-1].split(
        "fi", 1
    )[0]
    # Help may still mention multi-arch override examples; the push default branch must not.
    default_block = text.split("if [[ -z \"${PLATFORMS}\" ]]; then", 1)[1].split("\nfi", 1)[0]
    assert "linux/arm64" not in default_block
    assert 'PLATFORMS="linux/amd64"' in default_block


def test_d03_host_network_is_opt_in() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_DOCKER_BUILD_HOST_NETWORK" in text
    assert "HOST_NETWORK=" in text
    # Unconditional host networking must be gone from the buildx invocation.
    build_tail = text.rsplit("docker buildx build", 1)[-1]
    assert "--network host" not in build_tail or "NETWORK_ARGS" in text
    assert "--allow network.host" not in build_tail.split("NETWORK_ARGS")[0]
    assert 'NETWORK_ARGS+=(--allow network.host --network host)' in text
