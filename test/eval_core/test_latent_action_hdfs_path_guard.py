from __future__ import annotations

import pytest

from worldfoundry.base_models.perception_core.action_recognition.latent_action.backbones import (
    _rewrite_legacy_uri,
    _source,
)


def test_hdfs_uri_refused_without_explicit_allow() -> None:
    with pytest.raises(ValueError, match="WORLDFOUNDRY_ALLOW_HDFS_MNT_REWRITE"):
        _rewrite_legacy_uri("hdfs:///checkpoints/dino", env={})


def test_hdfs_uri_rewrite_opt_in_with_custom_prefix() -> None:
    rewritten = _rewrite_legacy_uri(
        "hdfs:///checkpoints/dino",
        env={
            "WORLDFOUNDRY_ALLOW_HDFS_MNT_REWRITE": "1",
            "WORLDFOUNDRY_HDFS_LOCAL_PREFIX": "/data/hdfs",
        },
    )
    assert rewritten == "/data/hdfs/checkpoints/dino"


def test_source_passes_through_local_legacy_path() -> None:
    resolved = _source(
        "DINO_V2_PATH",
        "dinov2_base",
        "dinov2_base_model_dir",
        "facebook/dinov2-base",
        env={"DINO_V2_PATH": "/opt/models/dinov2"},
    )
    assert resolved == "/opt/models/dinov2"
