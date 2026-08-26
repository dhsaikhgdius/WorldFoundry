"""CA-03: download_to_cache empty-hit and Content-Length guards."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_download_to_cache_rejects_empty_hit_and_length_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.io.disk import CACHE_MIN_FREE_ENV
    from worldfoundry.core.io import download as download_mod

    monkeypatch.setenv(CACHE_MIN_FREE_ENV, "0")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    empty_hit = cache_dir / "payload.bin"
    empty_hit.write_bytes(b"")

    class _Resp:
        def __init__(self, body: bytes, length: str | None) -> None:
            self._body = body
            self.headers = {} if length is None else {"Content-Length": length}

        def read(self, size: int = -1) -> bytes:
            if not self._body:
                return b""
            if size < 0:
                chunk, self._body = self._body, b""
                return chunk
            chunk, self._body = self._body[:size], self._body[size:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        download_mod.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Resp(b"abc", "3"),
    )
    cached = download_mod.download_to_cache("https://example.test/payload.bin", cache_dir=cache_dir)
    assert cached.read_bytes() == b"abc"

    monkeypatch.setattr(
        download_mod.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Resp(b"ab", "10"),
    )
    with pytest.raises(RuntimeError, match="incomplete download"):
        download_mod.download_to_cache(
            "https://example.test/other.bin",
            cache_dir=cache_dir,
            filename="other.bin",
        )
    assert not (cache_dir / "other.bin").exists()
