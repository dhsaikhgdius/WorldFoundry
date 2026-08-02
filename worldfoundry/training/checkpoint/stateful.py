"""Strict composition for named checkpointable training components."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

NAMED_STATEFUL_COLLECTION_SCHEMA = "worldfoundry-named-training-state"


class NamedStatefulCollection:
    """Expose several stateful objects as one inventory-checked component.

    The collection is intentionally structural: it neither owns nor mutates the
    component lifecycle. It only makes the exact set of scheduler, EMA, reward,
    or other algorithm states visible to the checkpoint layer.
    """

    def __init__(self, components: Mapping[str, object]) -> None:
        if not isinstance(components, Mapping):
            raise TypeError("named training-state components must be a mapping")
        normalized = {str(name): component for name, component in components.items()}
        if not normalized:
            raise ValueError("named training-state components cannot be empty")
        if any(not name.strip() for name in normalized):
            raise ValueError("named training-state component names cannot be empty")
        for name, component in normalized.items():
            if not callable(getattr(component, "state_dict", None)) or not callable(
                getattr(component, "load_state_dict", None)
            ):
                raise TypeError(f"named training-state component {name!r} must expose state_dict/load_state_dict")
        self.components = MappingProxyType(dict(sorted(normalized.items())))

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self.components)

    def state_dict(self) -> dict[str, object]:
        states: dict[str, object] = {}
        for name, component in self.components.items():
            state = component.state_dict()
            if not isinstance(state, Mapping):
                raise TypeError(f"named training-state component {name!r} returned a non-mapping state")
            states[name] = dict(state)
        return {
            "schema": NAMED_STATEFUL_COLLECTION_SCHEMA,
            "component_names": self.component_names,
            "components": states,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schema",
            "component_names",
            "components",
        }:
            raise ValueError("named training-state fields differ from the active schema")
        if state_dict["schema"] != NAMED_STATEFUL_COLLECTION_SCHEMA:
            raise ValueError(f"unsupported named training-state schema: {state_dict['schema']!r}")
        names = state_dict["component_names"]
        if not isinstance(names, (tuple, list)) or tuple(names) != self.component_names:
            raise ValueError("saved named training-state inventory differs from the active stack")
        states = state_dict["components"]
        if not isinstance(states, Mapping) or tuple(sorted(states)) != self.component_names:
            raise ValueError("saved named training-state components are incomplete")
        for name, component in self.components.items():
            component_state = states[name]
            if not isinstance(component_state, Mapping):
                raise ValueError(f"saved named training-state component {name!r} is invalid")
            component.load_state_dict(component_state)


__all__ = [
    "NAMED_STATEFUL_COLLECTION_SCHEMA",
    "NamedStatefulCollection",
]
