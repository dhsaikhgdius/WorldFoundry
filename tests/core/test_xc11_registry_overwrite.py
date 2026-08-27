"""XC-11: the legacy class registry refuses silent overwrites.

Guards the fix from plan/code_review/12_cross_cutting.md [XC-11]: the
``_CLASS_REGISTRY`` in :mod:`worldfoundry.core.io.config_utils` used to let a
later registration silently replace an earlier one under the same name
(last-write-wins). Now the first registration wins: re-registering the exact
same object is a no-op, while binding an already-registered name to a
*different* callable raises ``ValueError``. The dead ``make_registry_metaclass``
factory (zero callers) was removed from
``worldfoundry.core.utils.functional_utils`` in the same change.
"""

from __future__ import annotations

import pytest

from worldfoundry.core.io import config_utils


@pytest.fixture()
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Swap in an empty registry so tests never touch global state."""
    registry: dict = {}
    monkeypatch.setattr(config_utils, "_CLASS_REGISTRY", registry)
    return registry


class _WidgetA:
    pass


class _WidgetB:
    pass


def test_first_registration_succeeds(isolated_registry: dict) -> None:
    config_utils.register_class(_WidgetA)
    assert isolated_registry["_WidgetA"] is _WidgetA
    assert config_utils.get_class("_WidgetA") is _WidgetA


def test_same_class_reregistration_is_noop(isolated_registry: dict) -> None:
    config_utils.register_class(_WidgetA)
    config_utils.register_class(_WidgetA)  # must not raise
    assert isolated_registry["_WidgetA"] is _WidgetA


def test_different_class_same_name_raises(isolated_registry: dict) -> None:
    config_utils.register_class(_WidgetA)
    impostor = type("_WidgetA", (), {})
    with pytest.raises(ValueError, match="already registered"):
        config_utils.register_class(impostor)
    # the original binding survives the refused overwrite
    assert isolated_registry["_WidgetA"] is _WidgetA


def test_alias_conflict_raises(isolated_registry: dict) -> None:
    config_utils.register_class(alias=["widget"])(_WidgetA)
    assert isolated_registry["widget"] is _WidgetA
    with pytest.raises(ValueError, match="already registered"):
        config_utils.register_class(alias=["widget"])(_WidgetB)


def test_register_callable_follows_same_rules(isolated_registry: dict) -> None:
    config_utils.register_callable("factory", _WidgetA)
    config_utils.register_callable("factory", _WidgetA)  # same object: no-op
    with pytest.raises(ValueError, match="already registered"):
        config_utils.register_callable("factory", _WidgetB)


def test_make_registry_metaclass_removed() -> None:
    from worldfoundry.core import utils as core_utils
    from worldfoundry.core.utils import functional_utils

    assert not hasattr(functional_utils, "make_registry_metaclass")
    assert "make_registry_metaclass" not in core_utils.__all__
