"""D-06: .dockerignore uses **/ prefixes and excludes large latent COPY trees."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def test_d06_dockerignore_uses_globstar_prefixes() -> None:
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    assert "**/data/" in text
    assert "**/cache/" in text
    assert "**/tmp/" in text
    assert "**/*.mp4" in text
    assert "**/*.ckpt" in text
    # Bare rooted patterns that miss nested paths should not be the only form.
    lines = {line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}
    assert "data/" not in lines
    assert "cache/" not in lines


def test_d06_dockerignore_excludes_latent_copy_trees() -> None:
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    assert "worldfoundry/data/test_cases/" in text
    assert "thirdparty/" in text
    assert "plan/" in text
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" in text
