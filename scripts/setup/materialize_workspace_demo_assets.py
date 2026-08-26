#!/usr/bin/env python3
"""Materialize Workspace demo inputs from pinned official source checkouts.

DA-05: every official-git demo asset carries an explicit revision (never ``HEAD``).
When ``expected_sha256`` is set on a pin, materialize/check fails on mismatch.
EXTERNAL dataset pins are documented in ``plan/da05_workspace_demo_assets.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - PyYAML is a core install dep
    raise SystemExit("PyYAML is required to load workspace_demo_asset_pins.yaml") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PINS_PATH = Path(__file__).resolve().parent / "workspace_demo_asset_pins.yaml"
DEFAULT_REPOS_ROOT = REPO_ROOT / ".upstream_sources"
DEFAULT_CKPT_ROOT = REPO_ROOT.parent / "ckpt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "worldfoundry" / "data" / "test_cases"


def load_demo_asset_pins(path: Path | None = None) -> dict[str, Any]:
    """Load the committed pin manifest (repos + assets)."""

    pins_path = Path(path or DEFAULT_PINS_PATH)
    payload = yaml.safe_load(pins_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"pin manifest must be a mapping: {pins_path}")
    repos = payload.get("repos")
    if not isinstance(repos, Mapping) or not repos:
        raise ValueError(f"pin manifest missing repos: {pins_path}")
    for name, spec in repos.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"repo pin {name!r} must be a mapping")
        revision = str(spec.get("revision") or "").strip()
        if not revision or revision.upper() == "HEAD":
            raise ValueError(f"repo pin {name!r} must set a concrete revision (not HEAD)")
        if not str(spec.get("remote") or "").strip():
            raise ValueError(f"repo pin {name!r} must set remote")
    assets = payload.get("repo_assets") or []
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"pin manifest missing repo_assets: {pins_path}")
    return dict(payload)


def iter_pinned_repo_assets(pins: Mapping[str, Any]) -> list[dict[str, str]]:
    """Flatten repo + historical assets into concrete pin rows."""

    repos = pins["repos"]
    rows: list[dict[str, str]] = []
    for item in list(pins.get("repo_assets") or []) + list(pins.get("historical_repo_assets") or []):
        if not isinstance(item, Mapping):
            raise ValueError(f"asset pin must be a mapping, got {item!r}")
        target = str(item["target"])
        repo_name = str(item["repo"])
        path = str(item["path"])
        repo_spec = repos[repo_name]
        revision = str(item.get("revision") or repo_spec["revision"]).strip()
        if not revision or revision.upper() == "HEAD":
            raise ValueError(f"asset {target!r} must pin a concrete revision (not HEAD)")
        rows.append(
            {
                "target": target,
                "repo": repo_name,
                "path": path,
                "revision": revision,
                "remote": str(repo_spec["remote"]),
                "expected_sha256": str(item.get("expected_sha256") or "").strip(),
            }
        )
    return rows


def _replace_target(target: Path, source: Path, force: bool) -> str:
    if target.exists() and not force:
        return "ready"
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists() or target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return "materialized"


def _repo_revision(repo: Path, revision: str = "HEAD") -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", revision], text=True).strip()


def _repo_remote(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip()


def _extract_repo_path(repo: Path, source_path: str, destination: Path, revision: str) -> Path:
    if not revision or revision.upper() == "HEAD":
        raise ValueError("extract requires a concrete revision pin (not HEAD)")
    archive_path = destination / "source.tar"
    with archive_path.open("wb") as archive:
        subprocess.run(
            ["git", "-C", str(repo), "archive", revision, source_path],
            stdout=archive,
            check=True,
        )
    extracted_root = destination / "tree"
    extracted_root.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted_root, filter="data")
    extracted = extracted_root / source_path
    if not extracted.exists():
        raise FileNotFoundError(f"git archive did not contain {source_path} at {revision}")
    return extracted


def _hash_path(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    size = 0
    for item in files:
        relative = item.relative_to(path).as_posix() if path.is_dir() else item.name
        digest.update(relative.encode())
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    return digest.hexdigest(), size, len(files)


def _record(target: Path, status: str, source: dict[str, str], *, expected_sha256: str = "") -> dict[str, object]:
    row: dict[str, object] = {"target": str(target), "status": status, "source": source}
    if expected_sha256:
        row["expected_sha256"] = expected_sha256
    if target.exists():
        sha256, size, file_count = _hash_path(target)
        row.update(sha256=sha256, size_bytes=size, file_count=file_count)
        if expected_sha256 and sha256 != expected_sha256:
            row["status"] = "sha256_mismatch"
        if target.is_file() and target.stat().st_size < 1024:
            text = target.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("version https://git-lfs.github.com/spec/v1"):
                row["status"] = "lfs_pointer"
    elif expected_sha256 and status in {"ready", "materialized"}:
        row["status"] = "missing"
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS_PATH)
    parser.add_argument("--repos-root", type=Path, default=DEFAULT_REPOS_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true", help="Only report current readiness.")
    parser.add_argument(
        "--require-sha256",
        action="store_true",
        help="Fail when a pin lacks expected_sha256 (strict DA-05 mode after hashes are recorded).",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pins = load_demo_asset_pins(args.pins)
    rows: list[dict[str, object]] = []

    for asset in iter_pinned_repo_assets(pins):
        if args.require_sha256 and not asset["expected_sha256"]:
            rows.append(
                {
                    "target": str(args.output_root / asset["target"]),
                    "status": "sha256_pin_missing",
                    "source": {
                        "kind": "official_git",
                        "repo": asset["repo"],
                        "path": asset["path"],
                        "revision": asset["revision"],
                        "remote": asset["remote"],
                    },
                    "error": "expected_sha256 is required under --require-sha256",
                }
            )
            continue
        target = args.output_root / asset["target"]
        repo = args.repos_root / asset["repo"]
        source_meta = {
            "kind": "official_git",
            "repo": asset["repo"],
            "path": asset["path"],
            "revision": asset["revision"],
            "remote": asset["remote"],
        }
        try:
            if repo.is_dir():
                source_meta["resolved_revision"] = _repo_revision(repo, asset["revision"])
                source_meta["checkout_remote"] = _repo_remote(repo)
            if args.check:
                status = "ready" if target.exists() else "missing"
            else:
                with tempfile.TemporaryDirectory(prefix="worldfoundry-demo-") as temp:
                    source = _extract_repo_path(repo, asset["path"], Path(temp), revision=asset["revision"])
                    status = _replace_target(target, source, args.force)
            rows.append(
                _record(
                    target,
                    status,
                    source_meta,
                    expected_sha256=asset["expected_sha256"],
                )
            )
        except (FileNotFoundError, subprocess.CalledProcessError, tarfile.TarError, ValueError) as exc:
            rows.append(
                {
                    "target": str(target),
                    "status": "source_missing",
                    "source": source_meta,
                    "error": str(exc),
                }
            )

    for item in pins.get("checkpoint_assets") or []:
        target_name = str(item["target"])
        source_name = str(item["path"])
        target = args.output_root / target_name
        source = args.ckpt_root / source_name
        source_meta = {"kind": "official_checkpoint_asset", "path": str(source)}
        expected = str(item.get("expected_sha256") or "").strip()
        if args.check:
            status = "ready" if target.exists() else "missing"
        elif source.exists():
            status = _replace_target(target, source, args.force)
        else:
            status = "source_missing"
        rows.append(_record(target, status, source_meta, expected_sha256=expected))

    for item in pins.get("external_assets") or []:
        target = args.output_root / str(item["target"])
        rows.append(
            _record(
                target,
                "ready" if target.exists() else "external_pending",
                {"kind": "official_dataset", "uri": str(item["uri"])},
                expected_sha256=str(item.get("expected_sha256") or "").strip(),
            )
        )

    summary = {
        "schema_version": "worldfoundry-workspace-demo-assets-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pins_path": str(Path(args.pins).resolve()),
        "output_root": str(args.output_root),
        "asset_count": len(rows),
        "ready_count": sum(row["status"] in {"ready", "materialized"} for row in rows),
        "pending_count": sum(row["status"] not in {"ready", "materialized"} for row in rows),
    }
    payload = {"summary": summary, "assets": rows}
    if not args.check:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / ".workspace_demo_assets.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['status']}\t{row['target']}")
        print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pending_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
