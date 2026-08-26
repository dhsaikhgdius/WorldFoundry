"""NK: native-kernels README states sdist/[all] boundary and inspect helper."""

from __future__ import annotations

from pathlib import Path

README = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "worldfoundry-native-kernels"
    / "README.md"
)


def test_native_kernels_readme_states_packaging_boundary() -> None:
    text = README.read_text(encoding="utf-8")
    assert "[all]" in text
    assert "packages/worldfoundry-native-kernels" in text
    assert "native_provider_status" in text
    assert "CPU-only" in text or "CPU-only hosts" in text
