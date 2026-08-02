"""Integrity checks for the public benchmark catalog inventory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    EXCLUDED_BENCHMARK_IDS,
    benchmark_runtime_profiles_by_id,
    formal_benchmark_ids,
)
from worldfoundry.evaluation.utils import BENCHMARK_RUNTIME_PROFILE_DIR, BENCHMARK_ZOO_DIR, REPO_ROOT, load_manifest

_RUNNER_SCRIPT_RE = re.compile(r"worldfoundry/evaluation/tasks/execution/[A-Za-z0-9_./-]+\.py")
_BENCHMARK_ID_ARG_RE = re.compile(r"--benchmark-id(?:=|\s+)([A-Za-z0-9_.-]+)")
_TABLE_BACKTICK_RE = re.compile(r"`([a-z0-9][a-z0-9_.-]*)`")
_DOC_EXTENSIONS = {".md", ".mdx", ".ts", ".tsx"}


@dataclass(frozen=True)
class BenchmarkInventoryIntegrityReport:
    benchmark_ids: tuple[str, ...]
    catalog_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    runtime_profile_ids: tuple[str, ...]
    missing_task_ids: tuple[str, ...]
    extra_task_ids: tuple[str, ...]
    missing_runtime_profile_ids: tuple[str, ...]
    extra_runtime_profile_ids: tuple[str, ...]
    runner_scripts: tuple[str, ...]
    missing_runner_scripts: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_task_ids
            or self.extra_task_ids
            or self.missing_runtime_profile_ids
            or self.extra_runtime_profile_ids
            or self.missing_runner_scripts
        )

    def failure_summary(self) -> str:
        parts = []
        for name in (
            "missing_task_ids",
            "extra_task_ids",
            "missing_runtime_profile_ids",
            "extra_runtime_profile_ids",
            "missing_runner_scripts",
        ):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={list(value)}")
        return "; ".join(parts) if parts else "ok"


@dataclass(frozen=True)
class DocsBenchmarkCoverageReport:
    benchmark_ids: tuple[str, ...]
    mentioned_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    unknown_refs: tuple[str, ...]
    excluded_refs: tuple[str, ...]
    docs_paths: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing_ids or self.unknown_refs or self.excluded_refs)

    def failure_summary(self) -> str:
        parts = []
        for name in ("missing_ids", "unknown_refs", "excluded_refs"):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={list(value)}")
        return "; ".join(parts) if parts else "ok"


def _as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = load_manifest(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _iter_yaml_files(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        return (root,)
    return tuple(sorted(path for path in root.glob("*.yaml") if path.is_file() and path.name != "_manifest.yaml"))


def _collect_task_ids(task_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in _iter_yaml_files(task_dir):
        benchmark_id = _load_yaml_mapping(path).get("benchmark")
        if benchmark_id:
            ids.add(str(benchmark_id))
    return ids


def _collect_runner_scripts(value: Any) -> set[str]:
    scripts: set[str] = set()
    if isinstance(value, str):
        scripts.update(_RUNNER_SCRIPT_RE.findall(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            scripts.update(_collect_runner_scripts(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            scripts.update(_collect_runner_scripts(item))
    return scripts


def _collect_runner_scripts_from_paths(paths: Iterable[Path]) -> set[str]:
    scripts: set[str] = set()
    for path in paths:
        scripts.update(_collect_runner_scripts(_load_yaml_mapping(path)))
    return scripts


def build_benchmark_inventory_integrity(
    *,
    catalog_dir: str | Path | None = None,
    task_dir: str | Path,
    runtime_profile_path: str | Path | None = None,
) -> BenchmarkInventoryIntegrityReport:
    catalog_root = Path(catalog_dir or BENCHMARK_ZOO_DIR)
    task_root = Path(task_dir)
    runtime_root = Path(runtime_profile_path or (BENCHMARK_RUNTIME_PROFILE_DIR / "official"))

    benchmark_ids = set(formal_benchmark_ids(catalog_root))
    task_ids = _collect_task_ids(task_root)
    runtime_profile_ids = set(benchmark_runtime_profiles_by_id(runtime_root))

    catalog_paths = [path for shard in ("video", "embodied") for path in _iter_yaml_files(catalog_root / shard)]
    task_paths = list(_iter_yaml_files(task_root))
    runtime_paths = list(_iter_yaml_files(runtime_root if runtime_root.is_dir() else runtime_root.parent))
    runner_scripts = _collect_runner_scripts_from_paths((*catalog_paths, *task_paths, *runtime_paths))
    missing_runner_scripts = tuple(sorted(script for script in runner_scripts if not (REPO_ROOT / script).is_file()))

    return BenchmarkInventoryIntegrityReport(
        benchmark_ids=tuple(sorted(benchmark_ids)),
        catalog_ids=tuple(sorted(benchmark_ids)),
        task_ids=tuple(sorted(task_ids)),
        runtime_profile_ids=tuple(sorted(runtime_profile_ids)),
        missing_task_ids=tuple(sorted(benchmark_ids - task_ids)),
        extra_task_ids=tuple(sorted(task_ids - benchmark_ids)),
        missing_runtime_profile_ids=tuple(sorted(benchmark_ids - runtime_profile_ids)),
        extra_runtime_profile_ids=tuple(sorted(runtime_profile_ids - benchmark_ids)),
        runner_scripts=tuple(sorted(runner_scripts)),
        missing_runner_scripts=missing_runner_scripts,
    )


def _default_docs_paths() -> tuple[Path, ...]:
    benchmark_hub_dir = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmark-hub"
    candidates = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.mdx",
        benchmark_hub_dir,
        REPO_ROOT / "docs" / "fumadocs" / "lib" / "benchmark-hub-data.ts",
        REPO_ROOT / "docs" / "fumadocs" / "components" / "benchmark-hub.tsx",
    )
    paths: list[Path] = []
    for path in candidates:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix in _DOC_EXTENSIONS
                )
            )
    return tuple(dict.fromkeys(paths))


def _iter_docs_paths(roots: Iterable[str | Path] | None) -> tuple[Path, ...]:
    if roots is None:
        return _default_docs_paths()
    paths: list[Path] = []
    for root_value in roots:
        root = Path(root_value)
        if root.is_file():
            paths.append(root)
            continue
        if root.is_dir():
            paths.extend(sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in _DOC_EXTENSIONS))
    return tuple(dict.fromkeys(paths))


def _explicit_doc_refs(path: Path, text: str) -> tuple[str, ...]:
    refs: list[str] = []
    for match in _BENCHMARK_ID_ARG_RE.finditer(text):
        value = match.group(1).strip()
        if value and not value.startswith("<"):
            refs.append(value)
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        first_cell = line.split("|", 2)[1] if "|" in line[1:] else line
        if " / " not in first_cell:
            continue
        refs.extend(_TABLE_BACKTICK_RE.findall(first_cell))
    return tuple(dict.fromkeys(refs))


def build_docs_benchmark_coverage(
    *,
    docs_roots: Iterable[str | Path] | None = None,
    benchmark_ids: Iterable[str] | None = None,
) -> DocsBenchmarkCoverageReport:
    ids = tuple(sorted(str(item) for item in (benchmark_ids or formal_benchmark_ids())))
    id_set = set(ids)
    excluded_ids = set(EXCLUDED_BENCHMARK_IDS)
    docs_paths = _iter_docs_paths(docs_roots)

    mentioned: set[str] = set()
    unknown_refs: list[str] = []
    excluded_refs: list[str] = []
    for path in docs_paths:
        text = path.read_text(encoding="utf-8")
        for benchmark_id in ids:
            if benchmark_id in text:
                mentioned.add(benchmark_id)
        for ref in _explicit_doc_refs(path, text):
            rel = _as_repo_relative(path)
            if ref in id_set:
                mentioned.add(ref)
            elif ref in excluded_ids:
                excluded_refs.append(f"{rel}:{ref}")
            else:
                unknown_refs.append(f"{rel}:{ref}")

    return DocsBenchmarkCoverageReport(
        benchmark_ids=ids,
        mentioned_ids=tuple(sorted(mentioned)),
        missing_ids=tuple(sorted(id_set - mentioned)),
        unknown_refs=tuple(sorted(dict.fromkeys(unknown_refs))),
        excluded_refs=tuple(sorted(dict.fromkeys(excluded_refs))),
        docs_paths=tuple(_as_repo_relative(path) for path in docs_paths),
    )
