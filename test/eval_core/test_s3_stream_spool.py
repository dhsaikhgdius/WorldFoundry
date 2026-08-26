"""CA-04: S3 create_stream uses SpooledTemporaryFile instead of BytesIO."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_s3_filesystem_source_uses_spooled_temporary_file() -> None:
    source = (REPO_ROOT / "worldfoundry" / "core" / "io" / "s3_filesystem.py").read_text(encoding="utf-8")
    assert "SpooledTemporaryFile" in source
    assert "S3_STREAM_SPOOL_MAX_BYTES = 64 * 1024 * 1024" in source
    assert "io.BytesIO()" not in source


@pytest.mark.unit
def test_s3_create_stream_spools_and_round_trips() -> None:
    pytest.importorskip("boto3")
    pytest.importorskip("torch")
    from worldfoundry.core.io.s3_filesystem import (
        S3_STREAM_SPOOL_MAX_BYTES,
        S3FileSystem,
        _s3_spooled_stream,
    )

    assert S3_STREAM_SPOOL_MAX_BYTES == 64 * 1024 * 1024
    spool = _s3_spooled_stream()
    assert isinstance(spool, tempfile.SpooledTemporaryFile)
    spool.close()

    class _Client:
        def __init__(self) -> None:
            self.uploaded: bytes | None = None

        def download_fileobj(self, bucket, key, stream) -> None:
            assert bucket == "bucket"
            assert key == "key.bin"
            stream.write(b"checkpoint-bytes")

        def upload_fileobj(self, stream, bucket, key) -> None:
            self.uploaded = stream.read()

    fs = object.__new__(S3FileSystem)
    client = _Client()
    fs.s3_client = client

    with fs.create_stream("s3://bucket/key.bin", "rb") as handle:
        assert isinstance(handle, tempfile.SpooledTemporaryFile)
        assert handle.read() == b"checkpoint-bytes"

    with fs.create_stream("s3://bucket/key.bin", "wb") as handle:
        assert isinstance(handle, tempfile.SpooledTemporaryFile)
        handle.write(b"upload-bytes")
    assert client.uploaded == b"upload-bytes"
