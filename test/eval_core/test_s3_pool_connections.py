"""CA-06: boto3 connection pool sized for sync_s3_dir_to_local workers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_s3_filesystem_sets_max_pool_connections(tmp_path: Path) -> None:
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    pytest.importorskip("torch")

    from botocore.config import Config

    from worldfoundry.core.io.s3_filesystem import DEFAULT_S3_MAX_POOL_CONNECTIONS, S3FileSystem

    creds = tmp_path / "creds.json"
    creds.write_text(json.dumps({"region_name": "us-east-1"}), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_client(service_name, **kwargs):
        captured["service_name"] = service_name
        captured["kwargs"] = kwargs
        return object()

    with patch("worldfoundry.core.io.s3_filesystem.boto3.client", side_effect=_fake_client):
        fs = S3FileSystem(str(creds), max_pool_connections=48)

    assert fs.max_pool_connections == 48
    assert captured["service_name"] == "s3"
    cfg = captured["kwargs"]["config"]
    assert isinstance(cfg, Config)
    assert cfg.max_pool_connections == 48

    with patch("worldfoundry.core.io.s3_filesystem.boto3.client", side_effect=_fake_client):
        default_fs = S3FileSystem(str(creds))
    assert default_fs.max_pool_connections == DEFAULT_S3_MAX_POOL_CONNECTIONS
    assert captured["kwargs"]["config"].max_pool_connections == DEFAULT_S3_MAX_POOL_CONNECTIONS
