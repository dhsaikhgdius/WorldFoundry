"""Helpers for comparing WorldFoundry unified lockfile package pins (plan I-05)."""

from __future__ import annotations

import re

# Match ``name==version`` requirement lines; ignore editable / VCS / markers.
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._+-]*)==([^\s\\;]+)", re.MULTILINE)


def normalize_package_name(name: str) -> str:
    """Normalize a distribution name for pin comparison (PEP 503-ish)."""

    return re.sub(r"[-_.]+", "-", name.strip().lower())


def package_pins(lock_text: str) -> dict[str, str]:
    """Return ``{normalized_name: version}`` for ``==`` pins in *lock_text*."""

    pins: dict[str, str] = {}
    for match in _PIN.finditer(lock_text):
        pins[normalize_package_name(match.group(1))] = match.group(2).strip()
    return pins
