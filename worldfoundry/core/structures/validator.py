"""Descriptor base for reusable validated configuration fields."""

import itertools
import json
from abc import ABC, abstractmethod

_UNSET = object()


class Validator(ABC):
    def __init__(self, default=_UNSET, hidden: bool = False):
        self.default = default
        self.hidden = hidden

    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = getattr(obj, self.private_name, self.default)
        if value is _UNSET:
            raise ValueError(f"required parameter {self.private_name[1:]!r} has not been set")
        return value

    def __set__(self, obj, value):
        setattr(obj, self.private_name, self.validate(value))

    @abstractmethod
    def validate(self, value):
        raise NotImplementedError


class Bool(Validator):
    def __init__(self, default=_UNSET, hidden: bool = False, tooltip=None):
        super().__init__(default, hidden)
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, str):
            normalized = value.lower()
            if normalized not in {"true", "false", "1", "0"}:
                raise ValueError(f"Expected a boolean string, got {value!r}")
            return normalized in {"true", "1"}
        if isinstance(value, int):
            return value != 0
        if not isinstance(value, bool):
            raise TypeError(f"Expected bool, got {type(value).__name__}")
        return value

    def get_range_iterator(self):
        return [True, False]


class Int(Validator):
    def __init__(self, default=_UNSET, min=None, max=None, step=1, hidden: bool = False, tooltip=None):
        super().__init__(default, hidden)
        self.min = min
        self.max = max
        self.step = step
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, str):
            value = int(value)
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")
        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        default = 0 if self.default is _UNSET else int(self.default)
        lower = default if self.min is None else self.min
        upper = default + 100 if self.max is None else self.max
        return itertools.takewhile(lambda item: item <= upper, itertools.count(lower, self.step))


class Float(Validator):
    """Validate a bounded floating-point configuration value."""

    def __init__(self, default=0.0, min=None, max=None, step=0.5, hidden: bool = False, tooltip=None):
        super().__init__(default, hidden)
        self.min = min
        self.max = max
        self.step = step
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, (str, int)):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(f"Expected float, got {type(value).__name__}")
        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        lower = self.default if self.min is None else self.min
        upper = self.default if self.max is None else self.max
        return itertools.takewhile(lambda item: item <= upper, itertools.count(lower, self.step))


class OneOf(Validator):
    """Validate a value against an explicit set of choices."""

    def __init__(self, default, options, type_cast=None, hidden: bool = False, tooltip=None):
        super().__init__(default, hidden)
        self.options = set(options)
        self.type_cast = type_cast
        self.tooltip = tooltip

    def validate(self, value):
        if self.type_cast is not None:
            try:
                value = self.type_cast(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Expected {value!r} to be castable to {self.type_cast!r}") from exc
        if value not in self.options:
            raise ValueError(f"Expected {value!r} to be one of {self.options!r}")
        return value

    def get_range_iterator(self):
        return iter(self.options)


class String(Validator):
    def __init__(self, default=_UNSET, min=None, max=None, predicate=None, hidden: bool = False, tooltip=None):
        super().__init__(default, hidden)
        self.min = min
        self.max = max
        self.predicate = predicate
        self.tooltip = tooltip

    def validate(self, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"Expected str or None, got {type(value).__name__}")
        if self.min is not None and len(value) < self.min:
            raise ValueError(f"Expected {value!r} to have length at least {self.min}")
        if self.max is not None and len(value) > self.max:
            raise ValueError(f"Expected {value!r} to have length at most {self.max}")
        if self.predicate is not None and not self.predicate(value):
            raise ValueError(f"Predicate {self.predicate!r} rejected {value!r}")
        return value

    def get_range_iterator(self):
        return iter([self.default])


class JsonDict(Validator):
    def validate(self, value):
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        try:
            result = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Expected a JSON-encoded dictionary, got {value!r}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON dictionary, got {type(result).__name__}")
        return result


__all__ = ["Bool", "Float", "Int", "JsonDict", "OneOf", "String", "Validator", "_UNSET"]
