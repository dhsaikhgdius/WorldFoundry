"""Contract tests for the unified-install console-script wrappers (plan I-13).

scripts/setup/unified_install.sh writes hand-rolled bash wrappers into the
unified conda env because conda_install.sh never runs ``pip install -e .`` —
pip therefore never generates the ``[project.scripts]`` entry points inside
that env. The wrappers are the sole providers of the ``worldfoundry*``
commands there, so their name -> module map must not drift from
``[project.scripts]`` in pyproject.toml.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 only.
    tomllib = pytest.importorskip("tomli")


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_INSTALL = REPO_ROOT / "scripts" / "setup" / "unified_install.sh"

_WRAPPER_CALL = re.compile(
    r"^install_worldfoundry_wrapper\s+(?P<name>[\w.-]+)\s+(?P<module>[\w.]+)\s*$",
    re.MULTILINE,
)


def _wrapper_registrations() -> dict[str, str]:
    script = UNIFIED_INSTALL.read_text(encoding="utf-8")
    return {match["name"]: match["module"] for match in _WRAPPER_CALL.finditer(script)}


def _project_scripts() -> dict[str, str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    return payload["project"]["scripts"]


def test_wrapper_names_match_project_scripts() -> None:
    wrappers = _wrapper_registrations()
    scripts = _project_scripts()

    assert wrappers, "no install_worldfoundry_wrapper registrations found"
    assert set(wrappers) == set(scripts)


def test_wrapper_modules_match_entry_point_modules() -> None:
    wrappers = _wrapper_registrations()
    scripts = _project_scripts()

    for name, module in wrappers.items():
        entry_point = scripts[name]
        expected_module = entry_point.split(":", 1)[0]
        assert module == expected_module, (
            f"wrapper {name!r} execs `python -m {module}` but "
            f"[project.scripts] declares {entry_point!r}"
        )


def test_wrapper_modules_are_runnable_with_python_dash_m() -> None:
    for name, module in _wrapper_registrations().items():
        relative = Path(*module.split("."))
        as_module = REPO_ROOT / relative.with_suffix(".py")
        as_package_main = REPO_ROOT / relative / "__main__.py"
        assert as_module.is_file() or as_package_main.is_file(), (
            f"wrapper {name!r} execs `python -m {module}` but {module} is neither "
            f"a module file ({as_module}) nor a package with __main__.py "
            f"({as_package_main})"
        )


def test_wrapper_body_pins_repo_root_and_env_python() -> None:
    script = UNIFIED_INSTALL.read_text(encoding="utf-8")

    # The wrappers exist to pin the repo checkout and the env interpreter; if
    # either export or the exec form changes, the rationale comment and this
    # contract need a deliberate update.
    assert 'export WORLDFOUNDRY_REPO_ROOT="${ROOT}"' in script
    assert 'export PYTHONPATH="${ROOT}\\${PYTHONPATH:+:\\${PYTHONPATH}}"' in script
    assert 'exec "${ENV_PREFIX}/bin/python" -m ${module} "\\$@"' in script
