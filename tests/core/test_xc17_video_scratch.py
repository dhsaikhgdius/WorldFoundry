"""XC-17: materialize_video_input defaults to a cache-root scratch dir.

Guards the ``output_dir=None`` path of
``worldfoundry.core.io.video.materialize_video_input``: instead of a bare
``tempfile.mkdtemp()`` under ``/tmp``, the materialized video must land in a
``make_scratch_dir``-managed directory under
``${WORLDFOUNDRY_CACHE_DIR}/scratch/<date>/`` (removed best-effort at
interpreter exit). Callers that need the file to persist pass an explicit
``output_dir``, which must be honored unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from worldfoundry.core.io import video as video_module
from worldfoundry.core.io.video import materialize_video_input


@pytest.fixture(autouse=True)
def _isolated_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "wf_cache"
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("WORLDFOUNDRY_HOME", raising=False)
    return cache_root


@pytest.fixture(autouse=True)
def _stub_video_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Encoding real MP4s needs imageio/ffmpeg; these tests only assert where
    # the output lands, so write a placeholder file instead.
    def fake_save(frames, output_path, fps=16, **kwargs):
        Path(output_path).write_bytes(b"video-bytes")

    monkeypatch.setattr(video_module, "save_video_frames", fake_save)


def _frames():
    import numpy as np

    return np.zeros((2, 4, 4, 3), dtype=np.uint8)


class TestMaterializeVideoInputScratchDefault:
    def test_default_output_lands_under_cache_scratch(self, _isolated_cache_root: Path) -> None:
        result = Path(materialize_video_input(_frames()))
        assert result.is_file()
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        scratch_day_root = _isolated_cache_root / "scratch" / today
        assert result.name == "input.mp4"
        # materialize_video_input resolves its output path, so compare resolved forms.
        assert result.parent.parent == scratch_day_root.resolve()
        assert result.parent.name.startswith("worldfoundry_video_")

    def test_default_scratch_dir_registered_for_exit_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from worldfoundry.core.io import scratch as scratch_module

        registered: list[tuple] = []
        monkeypatch.setattr(
            scratch_module.atexit, "register", lambda func, *args, **kwargs: registered.append((func, args, kwargs))
        )
        result = Path(materialize_video_input(_frames()))
        assert len(registered) == 1
        _, args, _ = registered[0]
        assert Path(args[0]).resolve() == result.parent

    def test_explicit_output_dir_is_honored(self, tmp_path: Path, _isolated_cache_root: Path) -> None:
        persistent = tmp_path / "persistent_out"
        persistent.mkdir()
        result = Path(materialize_video_input(_frames(), output_dir=str(persistent), filename="clip.mp4"))
        assert result == persistent.resolve() / "clip.mp4"
        assert result.is_file()
        assert not (_isolated_cache_root / "scratch").exists()

    def test_existing_local_file_is_returned_without_materializing(
        self, tmp_path: Path, _isolated_cache_root: Path
    ) -> None:
        existing = tmp_path / "already_there.mp4"
        existing.write_bytes(b"existing")
        result = materialize_video_input(str(existing))
        assert result == str(existing.resolve())
        assert not (_isolated_cache_root / "scratch").exists()
