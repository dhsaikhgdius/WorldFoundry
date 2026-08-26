"""XC-9: kernel autotune boolean env switches use unambiguous ``_ENABLED`` names.

Guards the fix from plan/code_review/12_cross_cutting.md [XC-9]: the boolean
switches ``WORLDFOUNDRY_KERNEL_AUTOTUNE`` and
``WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE`` looked like near-duplicates of the path
variable ``WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_DIR``. Each switch now reads a
``..._ENABLED`` name first, while the legacy name remains a working fallback.
"""

from __future__ import annotations

import pytest

from worldfoundry.core.kernels.autotune_cache import persistent_autotune_cache_enabled
from worldfoundry.core.kernels.registry import kernel_autotune_enabled

_ALL_NAMES = (
    "WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED",
    "WORLDFOUNDRY_KERNEL_AUTOTUNE",
    "WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED",
    "WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE",
)


@pytest.fixture(autouse=True)
def _clean_autotune_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ALL_NAMES:
        monkeypatch.delenv(name, raising=False)


class TestKernelAutotuneEnabled:
    def test_default_is_disabled(self) -> None:
        assert kernel_autotune_enabled() is False

    def test_new_name_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "1")
        assert kernel_autotune_enabled() is True

    def test_legacy_name_still_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE", "1")
        assert kernel_autotune_enabled() is True

    def test_new_name_wins_when_disabling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "0")
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE", "1")
        assert kernel_autotune_enabled() is False

    def test_new_name_wins_when_enabling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "true")
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE", "0")
        assert kernel_autotune_enabled() is True

    def test_blank_new_name_falls_back_to_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "   ")
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE", "on")
        assert kernel_autotune_enabled() is True

    def test_values_are_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "TRUE")
        assert kernel_autotune_enabled() is True


class TestPersistentAutotuneCacheEnabled:
    def test_default_is_enabled(self) -> None:
        assert persistent_autotune_cache_enabled() is True

    def test_new_name_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED", "0")
        assert persistent_autotune_cache_enabled() is False

    def test_legacy_name_still_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE", "off")
        assert persistent_autotune_cache_enabled() is False

    def test_new_name_wins_when_enabling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED", "1")
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE", "0")
        assert persistent_autotune_cache_enabled() is True

    def test_new_name_wins_when_disabling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED", "false")
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE", "1")
        assert persistent_autotune_cache_enabled() is False

    def test_blank_new_name_falls_back_to_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED", "   ")
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE", "no")
        assert persistent_autotune_cache_enabled() is False
