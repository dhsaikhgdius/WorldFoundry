from __future__ import annotations

from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_synthesis import LingBotSynthesis


def test_fast_model_ref_resolves_base_sibling_even_with_fast_load_kwarg(tmp_path) -> None:
    hfd_root = tmp_path / "cache" / "hfd"
    base_dir = hfd_root / "robbyant--lingbot-world-base-cam"
    requested_fast_dir = hfd_root / "lingbot-world-fast"
    override_fast_dir = tmp_path / "override-fast"
    base_dir.mkdir(parents=True)
    requested_fast_dir.mkdir(parents=True)
    override_fast_dir.mkdir(parents=True)

    resolved_base, resolved_fast = LingBotSynthesis._resolve_model_paths(
        pretrained_model_path=str(requested_fast_dir),
        fast_model_path=str(override_fast_dir),
        use_fast=True,
        rank=0,
    )

    assert resolved_base == str(base_dir)
    assert resolved_fast == str(requested_fast_dir)
