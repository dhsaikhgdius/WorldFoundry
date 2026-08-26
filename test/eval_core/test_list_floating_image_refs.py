"""Tests for scripts/embodied/list_floating_image_refs.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "embodied" / "list_floating_image_refs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("list_floating_image_refs", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_image_ref_is_floating_helpers():
    mod = _load_module()
    assert mod.image_ref_is_floating("ghcr.io/example/img:latest")
    assert mod.image_ref_is_floating("ghcr.io/example/img")
    assert not mod.image_ref_is_floating("ghcr.io/example/img@sha256:" + ("a" * 64))
    assert not mod.image_ref_is_floating("ghcr.io/example/img:v1.2.3")


def test_collect_floating_refs_on_official_profiles():
    mod = _load_module()
    profile_dir = REPO_ROOT / "worldfoundry/data/benchmarks/runtime_profiles/official"
    findings = mod.collect_floating_refs(profile_dir)
    # Official profiles still ship :latest; checker must surface them without inventing digests.
    assert findings, "expected at least one floating :latest ref in official profiles"
    assert all("latest" in item["image"] or ":" not in item["image"].rsplit("/", 1)[-1] for item in findings)
