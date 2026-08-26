"""Contracts for open-source infer repro gate docs + Makefile (tip-stack of #32)."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.utils import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_open_source_quickstart_documents_clean_clone_infer_contract() -> None:
    model_manifest = load_manifest(
        REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog" / "world_models" / "matrix-game-2.yaml"
    )
    repo_id = model_manifest["checkpoint"]["repos"][0]["id"]
    revision = model_manifest["checkpoint"]["repos"][0]["sha"]
    doc_paths = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "quickstart.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "quickstart.zh.mdx",
    )

    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        assert "matrix-game-2" in text
        assert repo_id in text
        assert revision in text
        assert "WORLDFOUNDRY_HFD_ROOT" in text
        assert "worldfoundry-eval zoo model-download" in text
        assert "--model-id matrix-game-2" in text
        assert '--cache-dir "${WORLDFOUNDRY_HFD_ROOT}"' in text
        assert "--check-local" in text
        assert "worldfoundry-eval zoo model-validate" in text
        assert "models--Skywork--Matrix-Game-2.0" in text
        assert "Skywork--Matrix-Game-2.0" in text
        assert "ln -s /shared" in text
        assert "make open-source-infer-repro" in text
        assert "OPEN_SOURCE_INFER_HFD_ROOT" in text
        assert "OPEN_SOURCE_INFER_STRICT_LOCAL=1" in text
        assert "bash scripts/inference/test_nav_video_gen.sh matrix-game-2" in text
        assert "scorecard.json" in text
        assert "models/checkpoints/hfd" in text


def test_makefile_exposes_open_source_infer_repro_gate() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "open-source-infer-repro:" in makefile
    assert "OPEN_SOURCE_INFER_MODEL ?= matrix-game-2" in makefile
    assert "OPEN_SOURCE_INFER_HFD_ROOT ?= $(RELEASE_HFD_ROOT)" in makefile
    assert "OPEN_SOURCE_INFER_STRICT_LOCAL ?= 0" in makefile
    assert "scripts/model_zoo/open_source_infer_repro.py" in makefile
    assert "$(WORLDFOUNDRY_EVAL) zoo model-download" in makefile
    assert "$(WORLDFOUNDRY_EVAL) zoo model-validate" in makefile
    assert "--model-id $(OPEN_SOURCE_INFER_MODEL)" in makefile
