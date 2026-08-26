#!/usr/bin/env python3
"""Packaging license-gate checker (code_review VI-16).

Wheel content is decided by ``[tool.setuptools].packages.find`` in
``pyproject.toml``; MANIFEST.in ``prune`` lines only affect the sdist. This
checker keeps the two in agreement so license-gated vendored trees cannot
silently ship in an Apache-2.0 wheel:

1. Dead-rule guard: every ``exclude`` pattern in pyproject must match at
   least one discoverable package. A dead pattern is a stale snapshot of a
   directory layout that no longer exists and hides real gaps.
2. License deny-list: the MANIFEST.in block that starts with the
   "License-gated upstream runtimes" comment is the single source of truth
   for trees that may exist only as ignored local checkouts. No package in
   the post-exclude wheel set may fall under any of those prefixes -- even
   if someone later adds an ``__init__.py`` that makes a gated tree
   discoverable.
3. Optional built-artifact audit (``--wheel``): assert no file inside a
   built wheel lives under a gated path. Audit wheels built from a clean
   tree: a stale ``build/`` directory or ``*.egg-info`` manifest from an
   earlier install can reintroduce files that the current configuration
   excludes (build_py copies incrementally and never deletes).

Run via ``make packaging-check`` or the eval_core release gate.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
MANIFEST_PATH = REPO_ROOT / "MANIFEST.in"
LICENSE_GATE_MARKER = "License-gated upstream runtimes"


def load_find_config(pyproject_path: Path = PYPROJECT_PATH) -> dict:
    """Return the ``[tool.setuptools].packages.find`` table."""
    with pyproject_path.open("rb") as handle:
        config = tomllib.load(handle)
    return config["tool"]["setuptools"]["packages"]["find"]


def discover_packages(find_config: dict, *, apply_exclude: bool) -> list[str]:
    """Run setuptools package discovery with the pyproject configuration.

    In pyproject.toml ``packages.find`` defaults to ``namespaces = true``:
    discovery walks every directory (no ``__init__.py`` required), which is
    the basis actual wheel builds use. Auditing with plain ``find_packages``
    would under-report by thousands of namespace packages.
    """
    from setuptools import find_namespace_packages, find_packages

    finder = (
        find_namespace_packages
        if find_config.get("namespaces", True)
        else find_packages
    )
    where = find_config.get("where", ["."])[0]
    include = find_config.get("include", ("*",))
    exclude = find_config.get("exclude", ()) if apply_exclude else ()
    return finder(where=str(REPO_ROOT / where), include=include, exclude=exclude)


def dead_exclude_patterns(all_packages: list[str], exclude: list[str]) -> list[str]:
    """Exclude patterns that match zero discoverable packages."""
    return [
        pattern
        for pattern in exclude
        if not any(fnmatch.fnmatchcase(name, pattern) for name in all_packages)
    ]


def license_gated_paths(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Parse gated checkout paths from the MANIFEST.in license-gate block.

    The block starts at the comment containing ``License-gated upstream
    runtimes`` and ends at the first blank line. Only ``prune`` entries under
    ``worldfoundry/`` are package trees; ``thirdparty/`` entries are native
    extension checkouts with no importable package name.
    """
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    paths: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if LICENSE_GATE_MARKER in stripped:
            in_block = True
            continue
        if in_block:
            if not stripped:
                break
            if stripped.startswith("prune worldfoundry/"):
                paths.append(stripped.removeprefix("prune ").rstrip("/"))
    if not paths:
        raise RuntimeError(
            f"no license-gated prune entries found in {manifest_path}; "
            f"the '{LICENSE_GATE_MARKER}' block was moved or renamed"
        )
    return paths


def license_gated_prefixes(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Gated checkout paths as dotted package prefixes."""
    return [path.replace("/", ".") for path in license_gated_paths(manifest_path)]


def leaked_packages(kept_packages: list[str], prefixes: list[str]) -> list[str]:
    """Packages in the wheel set that fall under a gated prefix."""
    return sorted(
        name
        for name in kept_packages
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def audit_wheel(wheel_path: Path, gated_paths: list[str]) -> list[str]:
    """File entries inside a built wheel that live under a gated path."""
    dir_prefixes = tuple(path + "/" for path in gated_paths)
    with zipfile.ZipFile(wheel_path) as wheel:
        return sorted(
            entry
            for entry in wheel.namelist()
            if entry.startswith(dir_prefixes)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="additionally audit a built .whl file for gated paths",
    )
    args = parser.parse_args(argv)

    find_config = load_find_config()
    exclude = list(find_config.get("exclude", ()))
    all_packages = discover_packages(find_config, apply_exclude=False)
    kept_packages = discover_packages(find_config, apply_exclude=True)
    gated_paths = license_gated_paths()
    prefixes = [path.replace("/", ".") for path in gated_paths]

    failures = 0

    dead = dead_exclude_patterns(all_packages, exclude)
    if dead:
        failures += 1
        print(f"FAIL: {len(dead)} dead exclude pattern(s) in pyproject.toml:")
        for pattern in dead:
            print(f"  {pattern}")

    leaks = leaked_packages(kept_packages, prefixes)
    if leaks:
        failures += 1
        print(f"FAIL: {len(leaks)} license-gated package(s) in the wheel set:")
        for name in leaks:
            print(f"  {name}")

    if args.wheel is not None:
        wheel_leaks = audit_wheel(args.wheel, gated_paths)
        if wheel_leaks:
            failures += 1
            print(
                f"FAIL: {len(wheel_leaks)} license-gated file(s) "
                f"inside {args.wheel}:"
            )
            for entry in wheel_leaks[:50]:
                print(f"  {entry}")
            if len(wheel_leaks) > 50:
                print(f"  ... and {len(wheel_leaks) - 50} more")

    if failures:
        return 1

    print(
        "packaging license gate OK: "
        f"{len(all_packages)} discoverable packages, "
        f"{len(kept_packages)} in wheel set, "
        f"{len(exclude)} live exclude rules, "
        f"{len(prefixes)} gated prefixes clean"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
