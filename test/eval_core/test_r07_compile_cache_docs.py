"""R-07 docs: compile cache NFS risk + autotune remote-cache opt-in."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EN = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "environments.mdx"
ZH = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "environments.zh.mdx"


def test_r07_environments_documents_compile_cache() -> None:
    en = EN.read_text(encoding="utf-8")
    zh = ZH.read_text(encoding="utf-8")
    assert "### Compiler caches" in en
    assert "### 编译器缓存" in zh
    for text in (en, zh):
        assert "WORLDFOUNDRY_COMPILE_CACHE_DIR" in text
        assert "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE" in text
        assert "NFS" in text or "共享" in text
