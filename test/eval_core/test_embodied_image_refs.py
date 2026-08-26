from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.embodied.docker_runner import build_docker_run_command
from worldfoundry.evaluation.tasks.embodied.image_refs import (
    apply_digest,
    assert_image_ref_pinned,
    image_ref_is_floating,
    resolve_docker_image,
)


def test_image_ref_is_floating_detects_latest_and_untagged() -> None:
    assert image_ref_is_floating("ghcr.io/allenai/vla-evaluation-harness/libero:latest")
    assert image_ref_is_floating("example/bench")
    assert not image_ref_is_floating("example/bench:1.2.3")
    assert not image_ref_is_floating(
        "example/bench@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )


def test_resolve_docker_image_applies_digest_over_latest_tag() -> None:
    digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    resolved = resolve_docker_image(
        {"image": "ghcr.io/example/libero:latest", "digest": digest},
        require_pinned=True,
    )
    assert resolved == apply_digest("ghcr.io/example/libero:latest", digest)
    assert resolved.endswith(f"@sha256:{digest}")
    assert not image_ref_is_floating(resolved)


def test_assert_image_ref_pinned_raises_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_EMBODIED_REQUIRE_PINNED_IMAGES", raising=False)
    assert_image_ref_pinned("example/bench:latest", require=False)
    with pytest.raises(ValueError, match="refusing floating"):
        assert_image_ref_pinned("example/bench:latest", require=True)


def test_build_docker_run_command_uses_digest_resolved_image(tmp_path) -> None:
    digest = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    docker_config_path = tmp_path / "eval_config.yaml"
    docker_config_path.write_text("id: test\n", encoding="utf-8")
    cmd = build_docker_run_command(
        {"docker": {"image": "example/bench:latest", "digest": digest, "runtime": "nvidia"}},
        docker_config_path=docker_config_path,
        output_dir=tmp_path / "out",
    )
    assert f"example/bench@sha256:{digest}" in cmd


def test_official_calvin_profile_pins_real_ghcr_digest() -> None:
    """D-01: at least one official harness profile carries a registry-resolved digest."""

    import json
    from pathlib import Path

    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    profile_path = repo_root / "worldfoundry/data/benchmarks/runtime_profiles/official/calvin.yaml"
    digest_map_path = repo_root / "worldfoundry/data/benchmarks/runtime_profiles/official/docker_image_digests.json"

    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    docker = payload["docker"]
    assert docker["digest"].startswith("sha256:")
    assert len(docker["digest"]) == len("sha256:") + 64
    resolved = resolve_docker_image(docker, require_pinned=True)
    assert resolved.endswith("@" + docker["digest"])
    assert not image_ref_is_floating(resolved)
    assert not image_ref_is_floating(str(docker["source_image"]))

    digest_map = json.loads(digest_map_path.read_text(encoding="utf-8"))
    mapped = digest_map["images"]["ghcr.io/allenai/vla-evaluation-harness/calvin:latest"]["digest"]
    assert mapped == docker["digest"]
