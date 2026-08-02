from types import SimpleNamespace

from worldfoundry.core.io import video


def test_ffmpeg_resolver_uses_imageio_bundle_when_system_binary_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(video.shutil, "which", lambda _name: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "/opt/imageio/ffmpeg"),
    )

    assert video._resolve_ffmpeg_executable() == "/opt/imageio/ffmpeg"
