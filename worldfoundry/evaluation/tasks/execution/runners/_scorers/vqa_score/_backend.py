"""In-tree VQAScore backend (vendored t2v_metrics VQA path)."""

from __future__ import annotations

import shutil

from worldfoundry.runtime.jobs import run_bounded_command

_FFMPEG_CHECKED = False

# Version probe should be instant; the timeout guards against a hung binary.
_FFMPEG_PROBE_TIMEOUT_SECONDS = 60

_FFMPEG_INSTALL_HINT = (
    "ffmpeg is a required system requirement but not found. Install with:\n"
    "conda install ffmpeg=6.1.2 -c conda-forge\n"
    "or visit: https://ffmpeg.org/download.html"
)


def ensure_ffmpeg() -> None:
    """Validate ffmpeg availability before video scoring paths run."""
    global _FFMPEG_CHECKED
    if _FFMPEG_CHECKED:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(_FFMPEG_INSTALL_HINT)
    try:
        probe = run_bounded_command(("ffmpeg", "-version"), timeout=_FFMPEG_PROBE_TIMEOUT_SECONDS)
    except OSError as exc:
        raise RuntimeError(_FFMPEG_INSTALL_HINT) from exc
    # A timed-out probe surfaces as returncode 124 after a process-group kill,
    # so hung binaries map onto the same install hint as failing ones.
    if probe["returncode"] != 0:
        raise RuntimeError(_FFMPEG_INSTALL_HINT)
    _FFMPEG_CHECKED = True


__all__ = ["ensure_ffmpeg"]
