#!/usr/bin/env python3
"""Open-source inference reproducibility gate for clean-clone / shared HFD layouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldfoundry.evaluation.utils import load_manifest  # noqa: E402

DEFAULT_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
DOC_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "quickstart.mdx",
    REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "quickstart.zh.mdx",
)


def _find_manifest(manifest_dir: Path, model_id: str) -> Path:
    matches = sorted(manifest_dir.rglob(f"{model_id}.yaml")) + sorted(manifest_dir.rglob(f"{model_id}.yml"))
    if not matches:
        matches = sorted(manifest_dir.rglob(f"{model_id}.json"))
    if not matches:
        raise FileNotFoundError(f"no manifest found for model_id={model_id!r} under {manifest_dir}")
    return matches[0]


def _repo_ids(manifest: dict[str, Any]) -> list[str]:
    repos = ((manifest.get("checkpoint") or {}).get("repos") or ())
    ids: list[str] = []
    for repo in repos:
        if isinstance(repo, dict) and repo.get("id"):
            ids.append(str(repo["id"]))
    if not ids:
        for value in manifest.get("checkpoints") or ():
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())
    return ids


def _hf_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def _direct_hfd_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / repo_id.replace("/", "--")


def _local_check(cache_dir: Path, repo_id: str) -> dict[str, Any]:
    hf_dir = _hf_cache_dir(cache_dir, repo_id)
    direct_dir = _direct_hfd_dir(cache_dir, repo_id)
    snapshots = hf_dir / "snapshots"
    ready = False
    layout = "missing"
    if snapshots.is_dir() and any(path.is_dir() for path in snapshots.iterdir()):
        ready = True
        layout = "hf_cache"
    elif direct_dir.is_dir() and any(path.is_file() for path in direct_dir.rglob("*")):
        ready = True
        layout = "direct_hfd"
    return {
        "repo_id": repo_id,
        "ok": ready,
        "ready": ready,
        "layout": layout,
        "hf_cache_dir": str(hf_dir),
        "direct_hfd_dir": str(direct_dir),
    }


def _docs_check(model_id: str, repo_ids: list[str], revision: str | None) -> dict[str, Any]:
    missing: list[str] = []
    required_snippets = [
        model_id,
        "WORLDFOUNDRY_HFD_ROOT",
        "worldfoundry-eval zoo model-download",
        "worldfoundry-eval zoo model-validate",
        "make open-source-infer-repro",
        "OPEN_SOURCE_INFER_HFD_ROOT",
        "OPEN_SOURCE_INFER_STRICT_LOCAL=1",
        "bash scripts/inference/test_nav_video_gen.sh matrix-game-2",
        "scorecard.json",
        "ln -s /shared",
    ]
    for repo_id in repo_ids:
        required_snippets.append(repo_id)
        required_snippets.append(repo_id.replace("/", "--"))
        required_snippets.append(f"models--{repo_id.replace('/', '--')}")
    if revision:
        required_snippets.append(revision)

    for path in DOC_PATHS:
        if not path.is_file():
            missing.append(f"missing doc: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in required_snippets:
            if snippet not in text:
                missing.append(f"{path.name}: missing {snippet!r}")
    return {"ok": not missing, "missing": missing, "paths": [str(path) for path in DOC_PATHS]}


def _download_plan(model_id: str, cache_dir: Path, repo_ids: list[str]) -> dict[str, Any]:
    commands = [
        [
            "worldfoundry-eval",
            "zoo",
            "model-download",
            "--model-id",
            model_id,
            "--cache-dir",
            str(cache_dir),
            "--check-local",
        ]
    ]
    return {
        "ok": bool(repo_ids),
        "repo_ids": list(repo_ids),
        "commands": commands,
    }


def _symlink_layouts(cache_dir: Path, repo_ids: list[str]) -> dict[str, Any]:
    layouts = []
    for repo_id in repo_ids:
        leaf = repo_id.replace("/", "--")
        layouts.append(
            {
                "repo_id": repo_id,
                "hf_cache": str(cache_dir / f"models--{leaf}"),
                "direct_hfd": str(cache_dir / leaf),
                "example_link": f"ln -s /shared/{leaf} {cache_dir / leaf}",
            }
        )
    return {"ok": True, "layouts": layouts}


def build_report(
    *,
    model_id: str,
    manifest_dir: Path,
    cache_dir: Path,
    output_dir: Path | None = None,
    strict_local: bool = False,
) -> dict[str, Any]:
    """Build a clean-clone open-source inference reproducibility report."""

    manifest_path = _find_manifest(Path(manifest_dir), model_id)
    manifest = load_manifest(manifest_path)
    repo_ids = _repo_ids(manifest)
    revision = None
    repos = ((manifest.get("checkpoint") or {}).get("repos") or ())
    if repos and isinstance(repos[0], dict):
        revision = repos[0].get("sha") or repos[0].get("revision")

    local_checks = [_local_check(Path(cache_dir), repo_id) for repo_id in repo_ids]
    local_ready = bool(local_checks) and all(item.get("ready") for item in local_checks)
    docs = _docs_check(model_id, repo_ids, str(revision) if revision else None)
    download_plan = _download_plan(model_id, Path(cache_dir), repo_ids)
    symlink_layouts = _symlink_layouts(Path(cache_dir), repo_ids)
    local_check = {
        "ok": local_ready,
        "checks": local_checks,
    }

    ok = docs["ok"] and download_plan["ok"] and symlink_layouts["ok"]
    if strict_local:
        ok = ok and local_ready
    else:
        # Clean-clone gate: docs + plan must pass even when cache is empty.
        ok = ok and True

    report: dict[str, Any] = {
        "schema_version": "worldfoundry-open-source-infer-repro-report",
        "model_id": model_id,
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_dir),
        "ok": ok,
        "local_ready": local_ready,
        "docs": docs,
        "download_plan": download_plan,
        "symlink_layouts": symlink_layouts,
        "local_check": local_check,
        "strict_local": bool(strict_local),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "open_source_infer_repro.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="matrix-game-2")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict-local", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        model_id=args.model_id,
        manifest_dir=args.manifest_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        strict_local=bool(args.strict_local),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"ok={report['ok']} local_ready={report['local_ready']} model_id={report['model_id']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
