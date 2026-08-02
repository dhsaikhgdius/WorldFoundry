from __future__ import annotations

from scripts.docs.generate_benchmark_subpages import (
    _catalog_path,
    _load_yaml,
    _render_page,
    _write_detail_page,
)


def test_existing_benchmark_detail_page_is_preserved_by_default(tmp_path) -> None:
    page = tmp_path / "example.mdx"
    original = "benchmark-specific setup and commands\n"
    page.write_text(original, encoding="utf-8")

    action = _write_detail_page(page, "generic generated content\n", overwrite_existing=False)

    assert action == "preserved"
    assert page.read_text(encoding="utf-8") == original


def test_existing_benchmark_detail_page_requires_explicit_overwrite(tmp_path) -> None:
    page = tmp_path / "example.mdx"
    page.write_text("hand-written\n", encoding="utf-8")

    action = _write_detail_page(page, "replacement\n", overwrite_existing=True)

    assert action == "updated"
    assert page.read_text(encoding="utf-8") == "replacement\n"


def test_normalizer_only_page_does_not_advertise_unsupported_official_run() -> None:
    catalog_path = _catalog_path("worldreasonbench")
    assert catalog_path is not None

    page = _render_page(_load_yaml(catalog_path), "worldreasonbench", locale="en")

    assert "`normalizer_only`" in page
    assert "--mode normalizer" in page
    assert "--mode official-run" not in page
    assert "cannot yet replay the full official protocol" in page


def test_generated_page_labels_only_real_environment_variables_as_env_vars() -> None:
    catalog_path = _catalog_path("physics-iq-verified")
    assert catalog_path is not None

    page = _render_page(_load_yaml(catalog_path), "physics-iq-verified", locale="en")

    assert "- `WORLDFOUNDRY_PHYSICS_IQ_VERIFIED_ROOT`" in page
    assert "- `WORLDFOUNDRY_GENERATED_ARTIFACT_DIR`" in page
    assert "- `Official Physics-IQ-Verified media directory" not in page
