"""DO: fumadocs content pages must ship paired English and Chinese MDX."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs"


def _english_pages() -> list[Path]:
    return sorted(p for p in DOCS_ROOT.rglob("*.mdx") if not p.name.endswith(".zh.mdx"))


def _chinese_pages() -> list[Path]:
    return sorted(DOCS_ROOT.rglob("*.zh.mdx"))


def test_every_english_mdx_has_zh_pair() -> None:
    missing = [
        str(page.relative_to(DOCS_ROOT))
        for page in _english_pages()
        if not page.with_name(f"{page.stem}.zh.mdx").is_file()
    ]
    assert not missing, "missing Chinese pairs:\n" + "\n".join(missing)


def test_every_zh_mdx_has_english_pair() -> None:
    missing = []
    for page in _chinese_pages():
        english_name = page.name[: -len(".zh.mdx")] + ".mdx"
        if not page.with_name(english_name).is_file():
            missing.append(str(page.relative_to(DOCS_ROOT)))
    assert not missing, "missing English pairs:\n" + "\n".join(missing)


def _non_separator_pages(pages: list[str]) -> list[str]:
    """Fumadocs ``---Section---`` headers may be localized; page ids must match."""
    return [page for page in pages if not (page.startswith("---") and page.endswith("---"))]


def test_meta_json_page_ids_match_zh_counterparts() -> None:
    """Sidebar meta.json / meta.zh.json page ids must stay in lockstep."""
    drifts: list[str] = []
    for meta_en in sorted(DOCS_ROOT.rglob("meta.json")):
        meta_zh = meta_en.with_name("meta.zh.json")
        assert meta_zh.is_file(), f"missing {meta_zh.relative_to(DOCS_ROOT)}"
        en_payload = json.loads(meta_en.read_text(encoding="utf-8"))
        zh_payload = json.loads(meta_zh.read_text(encoding="utf-8"))
        en_pages = _non_separator_pages(list(en_payload.get("pages") or []))
        zh_pages = _non_separator_pages(list(zh_payload.get("pages") or []))
        if en_pages != zh_pages:
            drifts.append(
                f"{meta_en.relative_to(DOCS_ROOT)} page ids != "
                f"{meta_zh.relative_to(DOCS_ROOT)} page ids\n"
                f"  only_en={sorted(set(en_pages) - set(zh_pages))}\n"
                f"  only_zh={sorted(set(zh_pages) - set(en_pages))}"
            )
    assert not drifts, "meta page-id drift:\n" + "\n".join(drifts)
