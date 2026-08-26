"""DS-05: direct-layout HFD revision pins are validated by path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.core.io.paths import resolve_local_hf_model_path


def test_direct_hfd_revision_pin_is_validated(tmp_path: Path) -> None:
    direct = tmp_path / "owner--model"
    direct.mkdir()
    (direct / "config.json").write_text("cfg", encoding="utf-8")
    meta = direct / ".hfd"
    meta.mkdir()
    (meta / "repo_metadata.json").write_text(
        '{"sha": "' + ("a" * 40) + '"}',
        encoding="utf-8",
    )
    env = {
        "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "checkpoints"),
        "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
    }

    resolved = resolve_local_hf_model_path(
        direct,
        required_files=("config.json",),
        revision="a" * 40,
        env=env,
    )
    assert resolved == direct.resolve()

    with pytest.raises(FileNotFoundError):
        resolve_local_hf_model_path(
            direct,
            required_files=("config.json",),
            revision="b" * 40,
            env=env,
        )


def test_direct_hfd_revision_pin_allows_unpinned_lookup(tmp_path: Path) -> None:
    direct = tmp_path / "owner--model"
    direct.mkdir()
    (direct / "config.json").write_text("cfg", encoding="utf-8")
    env = {
        "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "checkpoints"),
        "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
    }
    resolved = resolve_local_hf_model_path(
        direct,
        required_files=("config.json",),
        env=env,
    )
    assert resolved == direct.resolve()
