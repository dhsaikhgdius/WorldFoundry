"""Local asset discovery and manifest resolution for WorldFoundry benchmarks.

This module loads YAML manifests describing locally staged benchmark and model
assets, resolves ``$WORLDFOUNDRY_*`` path tokens inside manifest entries, and
produces :class:`LocalAsset` instances that report whether each asset path
exists on disk.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.manifests import load_manifest
from worldfoundry.core.io.paths import (
    package_data_root,
    project_root,
    resolve_worldfoundry_path,
)
from worldfoundry.core.io.paths import (
    worldfoundry_path_tokens as core_worldfoundry_path_tokens,
)

from .env import (
    EnvMapping,
    benchmark_repo_cache_root,
    resolve_artifact_dir,
    resolve_cache_dir,
    resolve_ckpt_dir,
    resolve_data_dir,
    resolve_hfd_root,
    resolve_model_dir,
)

# ── Constants ────────────────────────────────────────────────────────────────

LOCAL_ASSET_MANIFEST_ENV = "WORLDFOUNDRY_LOCAL_ASSET_MANIFEST"

# Same values evaluation.utils derives; resolved from core so the runtime
# layer never imports worldfoundry.evaluation (SA-10).
REPO_ROOT = project_root()
BENCHMARKS_DATA_ROOT = package_data_root() / "benchmarks"
# Keep shard names aligned with evaluation.tasks.catalog.benchmark_catalog
# DEFAULT_CATALOG_SHARD_DIRS (video / embodied) without importing evaluation.
CATALOG_SHARD_NAMES: tuple[str, ...] = ("video", "embodied")
CATALOG_MANIFEST_FILENAME = "_manifest.yaml"


def _path_is_ready(path: Path | None) -> bool:
    """Return whether a resolved asset path looks usable.

    Plain existence is not enough for file assets: interrupted downloads and
    copies commonly leave zero-byte files behind, which then fail much later
    inside model loading instead of at asset resolution time.
    """
    if path is None:
        return False
    try:
        if path.is_file():
            return path.stat().st_size > 0
        return path.exists()
    except OSError:
        return False


@dataclass(frozen=True)
class LocalAsset:
    """Describe one locally staged benchmark or model asset.

    Args:
        benchmark_id: Optional benchmark or integration id that owns the asset.
        asset_id: Stable asset id inside the benchmark group.
        kind: Asset kind, such as dataset, checkpoint, repo, manifest, or artifact.
        path: Resolved local path recorded by the manifest.
        canonical_path: Preferred path under the WorldFoundry root layout.
        status: Current path status computed at load time.
        ready: Whether the resolved path exists locally.
        metadata: Extra manifest fields preserved for consumers.
    """

    benchmark_id: str | None
    asset_id: str
    kind: str
    path: Path | None
    canonical_path: Path | None
    status: str
    ready: bool
    metadata: Mapping[str, Any]

    @classmethod
    def from_manifest_item(
        cls,
        item: Mapping[str, Any],
        *,
        benchmark_id: str | None = None,
        env: EnvMapping | None = None,
    ) -> "LocalAsset":
        """Build a local asset view from a manifest item.

        Args:
            item: Manifest asset mapping.
            benchmark_id: Optional parent benchmark id.
            env: Optional environment mapping used for path token expansion.
        """

        raw_env = item.get("env")
        if isinstance(raw_env, str) and raw_env.strip() and os.environ.get(raw_env.strip()):
            raw_path = os.environ[raw_env.strip()]
        else:
            raw_path = item.get("path") or item.get("local_path")
        raw_canonical_path = item.get("canonical_path")
        path = expand_worldfoundry_path(raw_path, env) if raw_path else None
        canonical_path = expand_worldfoundry_path(raw_canonical_path, env) if raw_canonical_path else None
        ready = _path_is_ready(path)
        status = "available" if ready else "missing"
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"id", "asset_id", "kind", "path", "local_path", "canonical_path", "status"}
        }
        return cls(
            benchmark_id=benchmark_id,
            asset_id=str(item.get("id") or item.get("asset_id") or "asset"),
            kind=str(item.get("kind") or "asset"),
            path=path,
            canonical_path=canonical_path,
            status=status,
            ready=ready,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resolved asset status for logs and diagnostics.

        Args:
            None.
        """

        payload: dict[str, Any] = {
            "benchmark_id": self.benchmark_id,
            "id": self.asset_id,
            "kind": self.kind,
            "path": str(self.path) if self.path is not None else None,
            "canonical_path": str(self.canonical_path) if self.canonical_path is not None else None,
            "status": self.status,
            "ready": self.ready,
        }
        payload.update(self.metadata)
        return payload


def _repo_root() -> Path:
    """Resolve the WorldFoundry repository root from the installed source tree.

    Args:
        None.
    """

    return REPO_ROOT


def worldfoundry_path_tokens(env: EnvMapping | None = None) -> dict[str, str]:
    """Return manifest path tokens backed by runtime environment helpers.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    environ = os.environ if env is None else env
    repo_root = _repo_root()
    tokens = core_worldfoundry_path_tokens(environ)
    tokens.update({
        "WORLDFOUNDRY_BENCH_ROOT": str(Path(environ.get("WORLDFOUNDRY_BENCH_ROOT") or repo_root).expanduser()),
        "WORLDFOUNDRY_REPO_ROOT": str(Path(environ.get("WORLDFOUNDRY_REPO_ROOT") or repo_root).expanduser()),
        "WORLDFOUNDRY_CACHE_DIR": str(resolve_cache_dir(environ)),
        "WORLDFOUNDRY_DATA_DIR": str(resolve_data_dir(environ)),
        "WORLDFOUNDRY_MODEL_DIR": str(resolve_model_dir(environ)),
        "WORLDFOUNDRY_ARTIFACT_DIR": str(resolve_artifact_dir(environ)),
        "WORLDFOUNDRY_CKPT_DIR": str(resolve_ckpt_dir(environ)),
        "WORLDFOUNDRY_HFD_ROOT": str(resolve_hfd_root(environ)),
        "WORLDFOUNDRY_HFD_DATASET_ROOT": str(
            Path(environ.get("WORLDFOUNDRY_HFD_DATASET_ROOT") or resolve_data_dir(environ)).expanduser()
        ),
        "WORLDFOUNDRY_BENCHMARK_REPO_ROOT": str(benchmark_repo_cache_root(environ)),
    })
    return tokens


def expand_worldfoundry_path(value: str | Path, env: EnvMapping | None = None) -> Path:
    """Expand a manifest path containing WorldFoundry environment tokens.

    Args:
        value: Path string or ``Path`` with optional ``$VAR`` or ``${VAR}`` tokens.
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    path = resolve_worldfoundry_path(value, worldfoundry_path_tokens(env))
    if path.is_absolute():
        return path
    return _repo_root() / path


def default_asset_manifest_candidates(env: EnvMapping | None = None) -> tuple[Path, ...]:
    """Return local asset manifest candidates in preferred lookup order.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    environ = os.environ if env is None else env
    repo_root = _repo_root()
    explicit = environ.get(LOCAL_ASSET_MANIFEST_ENV)
    candidates = [
        resolve_cache_dir(environ) / "manifests" / "local_assets_manifest.yaml",
        repo_root / "tmp" / "benchmark_zoo" / "local_assets_manifest.yaml",
        BENCHMARKS_DATA_ROOT / "local_assets_manifest.yaml",
        BENCHMARKS_DATA_ROOT / "local_assets.example.yaml",
    ]
    if explicit:
        candidates.insert(0, Path(explicit).expanduser())
    return tuple(candidates)


def resolve_asset_manifest_path(path: str | Path | None = None, env: EnvMapping | None = None) -> Path:
    """Resolve the local asset manifest path without touching benchmark runtimes.

    Args:
        path: Optional explicit manifest path.
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    if path is not None:
        return Path(path).expanduser()
    environ = os.environ if env is None else env
    explicit = environ.get(LOCAL_ASSET_MANIFEST_ENV)
    if explicit:
        return Path(explicit).expanduser()
    candidates = default_asset_manifest_candidates(env)
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    return existing or candidates[0]


def load_local_asset_manifest(path: str | Path | None = None, env: EnvMapping | None = None) -> dict[str, Any]:
    """Read the local benchmark/model asset manifest as YAML.

    Args:
        path: Optional explicit manifest path.
        env: Optional environment mapping used for default path resolution.
    """

    manifest_path = resolve_asset_manifest_path(path, env)
    payload = load_manifest(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("local asset manifest must be a YAML mapping")
    return payload


def iter_manifest_asset_items(manifest: Mapping[str, Any]) -> Iterator[tuple[str | None, Mapping[str, Any]]]:
    """Yield benchmark id and asset item pairs from a local asset manifest.

    Args:
        manifest: Loaded local asset manifest mapping.
    """

    root_assets = manifest.get("assets")
    if isinstance(root_assets, list):
        for item in root_assets:
            if isinstance(item, Mapping):
                yield None, item
    benchmarks = manifest.get("benchmarks")
    if isinstance(benchmarks, list):
        for benchmark in benchmarks:
            if not isinstance(benchmark, Mapping):
                continue
            benchmark_id = benchmark.get("id") or benchmark.get("benchmark_id")
            assets = benchmark.get("assets")
            if not isinstance(assets, list):
                continue
            for item in assets:
                if isinstance(item, Mapping):
                    yield str(benchmark_id) if benchmark_id is not None else None, item


def load_local_assets(path: str | Path | None = None, env: EnvMapping | None = None) -> tuple[LocalAsset, ...]:
    """Load local asset entries with resolved path status.

    Args:
        path: Optional explicit manifest path.
        env: Optional environment mapping used for path expansion.
    """

    manifest = load_local_asset_manifest(path, env)
    return tuple(
        LocalAsset.from_manifest_item(item, benchmark_id=benchmark_id, env=env)
        for benchmark_id, item in iter_manifest_asset_items(manifest)
    )


def iter_bundled_catalog_benchmark_ids(
    catalog_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return catalog benchmark ids under the bundled video/embodied shards.

    Args:
        catalog_root: Optional catalog root; defaults to ``BENCHMARKS_DATA_ROOT/catalog``.
    """

    root = Path(catalog_root) if catalog_root is not None else BENCHMARKS_DATA_ROOT / "catalog"
    ids: list[str] = []
    for shard in CATALOG_SHARD_NAMES:
        shard_dir = root / shard
        if not shard_dir.is_dir():
            continue
        for path in sorted(shard_dir.glob("*.yaml")):
            if path.name == CATALOG_MANIFEST_FILENAME or not path.is_file():
                continue
            ids.append(path.stem)
    return tuple(ids)


def local_asset_coverage_policy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the optional ``coverage`` policy block from a local asset manifest."""

    coverage = manifest.get("coverage")
    return dict(coverage) if isinstance(coverage, Mapping) else {}


def local_asset_manifest_benchmark_ids(manifest: Mapping[str, Any]) -> frozenset[str]:
    """Return benchmark ids that have at least one asset row in the manifest."""

    ids: set[str] = set()
    for benchmark_id, _item in iter_manifest_asset_items(manifest):
        if benchmark_id:
            ids.add(benchmark_id)
    benchmarks = manifest.get("benchmarks")
    if isinstance(benchmarks, list):
        for benchmark in benchmarks:
            if isinstance(benchmark, Mapping):
                bid = benchmark.get("id") or benchmark.get("benchmark_id")
                if bid is not None:
                    ids.add(str(bid))
    return frozenset(ids)


def resolve_local_asset_coverage_alias(
    benchmark_id: str,
    *,
    coverage: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Map a catalog id through ``coverage.aliases`` when present."""

    policy = coverage if coverage is not None else local_asset_coverage_policy(manifest or {})
    aliases = policy.get("aliases")
    if isinstance(aliases, Mapping) and benchmark_id in aliases:
        return str(aliases[benchmark_id])
    return benchmark_id


def uncovered_catalog_benchmark_ids(
    *,
    manifest: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
    env: EnvMapping | None = None,
    catalog_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return catalog ids missing from asset rows, aliases, and explicit exemptions.

    DA-07: gaps used to silently fall through the final asset fallback chain.
    Every catalog id must be listed under ``benchmarks``, ``coverage.aliases``,
    or ``coverage.exempt_ids``.
    """

    payload = manifest if manifest is not None else load_local_asset_manifest(path, env)
    policy = local_asset_coverage_policy(payload)
    listed = local_asset_manifest_benchmark_ids(payload)
    aliases = policy.get("aliases") if isinstance(policy.get("aliases"), Mapping) else {}
    exempt_raw = policy.get("exempt_ids") or ()
    exempt = {str(item) for item in exempt_raw} if isinstance(exempt_raw, (list, tuple, set, frozenset)) else set()

    gaps: list[str] = []
    for benchmark_id in iter_bundled_catalog_benchmark_ids(catalog_root):
        if benchmark_id in listed or benchmark_id in exempt:
            continue
        alias_target = None
        if isinstance(aliases, Mapping) and benchmark_id in aliases:
            alias_target = str(aliases[benchmark_id])
        if alias_target is not None and alias_target in listed:
            continue
        gaps.append(benchmark_id)
    return tuple(gaps)
