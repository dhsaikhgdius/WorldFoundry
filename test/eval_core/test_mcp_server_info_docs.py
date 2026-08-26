"""MC-01: server_info docs links must resolve to fumadocs content pages."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.mcp.tools.server_info import server_info_payload

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_ROOT = _REPO_ROOT / "docs" / "fumadocs" / "content" / "docs"


def _docs_path_exists(docs_href: str) -> Path | None:
    """Map ``/docs/...`` hrefs to ``content/docs/...`` mdx files."""

    assert docs_href.startswith("/docs/"), docs_href
    rel = docs_href[len("/docs/") :].strip("/")
    candidates = (
        _DOCS_ROOT / f"{rel}.mdx",
        _DOCS_ROOT / rel / "index.mdx",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def test_server_info_docs_links_resolve_to_fumadocs_pages() -> None:
    payload = server_info_payload()
    docs = payload["docs"]
    assert isinstance(docs, dict) and docs, "server_info must advertise docs links"

    missing: list[str] = []
    for key, href in sorted(docs.items()):
        assert isinstance(href, str) and href.startswith("/docs/"), (key, href)
        if _docs_path_exists(href) is None:
            missing.append(f"{key}={href}")

    assert not missing, "dead server_info docs links:\n" + "\n".join(missing)


def test_server_info_docs_cover_expected_surfaces() -> None:
    docs = server_info_payload()["docs"]
    for key in ("mcp", "cli", "evaluation_quickstart", "metrics", "agent_setup", "docker"):
        assert key in docs, f"missing docs key: {key}"
