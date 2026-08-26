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


def test_official_same_repo_profiles_promote_image_to_digest() -> None:
    """D-01 follow-up: when source_image is digest-pinned and shares the repo, image is not :latest."""

    import yaml
    from pathlib import Path

    from worldfoundry.evaluation.tasks.embodied.image_refs import (
        KNOWN_FLOATING_OFFICIAL_PROFILES,
        image_ref_is_floating,
        repository_name,
        resolve_docker_image,
    )

    root = Path(__file__).resolve().parents[2] / "worldfoundry/data/benchmarks/runtime_profiles/official"
    checked = 0
    for path in sorted(root.glob("*.yaml")):
        if path.stem in KNOWN_FLOATING_OFFICIAL_PROFILES:
            continue
        docker = yaml.safe_load(path.read_text(encoding="utf-8")).get("docker") or {}
        image = str(docker.get("image") or "")
        source = str(docker.get("source_image") or "")
        if not image or not source or "@sha256:" not in source:
            continue
        if repository_name(image) != repository_name(source):
            continue
        assert not image_ref_is_floating(image), path.name
        assert image == source, path.name
        resolved = resolve_docker_image(docker, require_pinned=True)
        assert not image_ref_is_floating(resolved), path.name
        checked += 1
    assert checked >= 12, checked


def test_known_floating_official_profiles_are_exact_allowlist() -> None:
    """D-01: only auth-gated / cross-repo mirrors may keep floating docker.image."""

    import yaml
    from pathlib import Path

    from worldfoundry.evaluation.tasks.embodied.image_refs import (
        AUTH_GATED_FLOATING_OFFICIAL_PROFILES,
        CROSS_REPO_FLOATING_MIRROR_PROFILES,
        KNOWN_FLOATING_OFFICIAL_PROFILES,
        image_ref_is_floating,
    )

    root = Path(__file__).resolve().parents[2] / "worldfoundry/data/benchmarks/runtime_profiles/official"
    found: set[str] = set()
    for path in sorted(root.glob("*.yaml")):
        docker = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("docker") or {}
        image = str(docker.get("image") or "").strip()
        source = str(docker.get("source_image") or "").strip()
        if not image:
            continue
        if image_ref_is_floating(image) or (source and image_ref_is_floating(source)):
            found.add(path.stem)
    assert found == set(KNOWN_FLOATING_OFFICIAL_PROFILES)
    assert AUTH_GATED_FLOATING_OFFICIAL_PROFILES.isdisjoint(CROSS_REPO_FLOATING_MIRROR_PROFILES)
    assert "libero" in CROSS_REPO_FLOATING_MIRROR_PROFILES
    assert "behavior1k" in AUTH_GATED_FLOATING_OFFICIAL_PROFILES

