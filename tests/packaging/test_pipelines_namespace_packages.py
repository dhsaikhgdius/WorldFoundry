from __future__ import annotations

from pathlib import Path

import tomllib
from setuptools import find_namespace_packages, find_packages
from setuptools.discovery import PEP420PackageFinder


def test_pyproject_enables_namespace_package_discovery() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    find = data["tool"]["setuptools"]["packages"]["find"]
    assert find.get("namespaces") is True


def test_pipelines_package_is_discovered_for_wheel() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    find = data["tool"]["setuptools"]["packages"]["find"]
    include = list(find.get("include", ["worldfoundry*"]))
    exclude = list(find.get("exclude", []))

    # setuptools' PEP 420 finder (what ``namespaces = true`` enables) must see
    # the pipelines tree so binding modules ship in the wheel.
    namespaced = PEP420PackageFinder.find(".", include=include, exclude=exclude)
    assert "worldfoundry.pipelines" in namespaced
    pipeline_children = [name for name in namespaced if name.startswith("worldfoundry.pipelines.")]
    assert pipeline_children, "expected at least one worldfoundry.pipelines.* binding package"

    # Root __init__.py also makes the classic finder see the package root.
    classic = find_packages(include=include, exclude=exclude)
    assert "worldfoundry.pipelines" in classic

    # Mirror the same discovery via find_namespace_packages for API coverage.
    assert "worldfoundry.pipelines" in find_namespace_packages(include=include, exclude=exclude)
