"""CA-05: S3 sync validation stamps skip warm HEAD/SHA256."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_s3_sync_validation_stamp_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tqdm")
    pytest.importorskip("boto3")
    from worldfoundry.core.io import s3_sync

    local = tmp_path / "asset.bin"
    local.write_bytes(b"payload-bytes")

    s3_sync._write_validation_stamp(
        str(local),
        content_length=local.stat().st_size,
        checksum_sha256="abc",
        etag='"etag"',
    )
    stamp = s3_sync._read_validation_stamp(str(local))
    assert stamp is not None
    assert s3_sync._validation_stamp_matches(str(local), stamp)

    local.write_bytes(b"changed")
    assert not s3_sync._validation_stamp_matches(str(local), stamp)
