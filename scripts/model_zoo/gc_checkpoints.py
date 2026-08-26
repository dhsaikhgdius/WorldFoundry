#!/usr/bin/env python3
"""List (and optionally delete) local checkpoint dirs not referenced by catalogs.

Default mode is dry-run: print candidates under the HFD / checkpoint root that
look like model checkout directories but are not mentioned by any catalog
manifest ``hf_repo_id`` / checkpoint ref. Pass ``--delete`` to remove them.

This does not invent digests or download anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.evaluation.utils import load_manifest  # noqa: E402
from worldfoundry.runtime.env import resolve_ckpt_dir, resolve_hfd_root  # noqa: E402

DEFAULT_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
MANIFEST_EXTENSIONS = (".json", ".yaml", ".yml")


def _hf_repo_id_from_value(value: Any) -> str | None:
    if isinstance(value, str) and "/" in value and not value.startswith(("http://", "https://", "file:")):
        return value.strip()
    if isinstance(value, dict):
        for key in ("hf_repo_id", "repo_id", "id"):
            got = _hf_repo_id_from_value(value.get(key))
            if got:
                return got
    return None


def _collect_repo_ids(data: dict[str, Any]) -> set[str]:
    found: set[str] = set()

    def add(value: Any) -> None:
        repo = _hf_repo_id_from_value(value)
        if repo:
            found.add(repo)

    add(data.get("hf_repo_id"))
    source = data.get("source")
    if isinstance(source, dict):
        add(source.get("hf_repo_id"))
    checkpoint = data.get("checkpoint")
    if isinstance(checkpoint, dict):
        add(checkpoint.get("hf_repo_id"))
        for repo in checkpoint.get("repos") or ():
            add(repo)
    for key in ("checkpoint_refs", "checkpoints"):
        for repo in data.get(key) or ():
            add(repo)
    return found


def referenced_repo_ids(manifest_dir: Path) -> set[str]:
    refs: set[str] = set()
    for path in sorted(manifest_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MANIFEST_EXTENSIONS:
            continue
        try:
            payload = load_manifest(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            refs.update(_collect_repo_ids(payload))
    return refs


def _local_name_for_repo(repo_id: str) -> str:
    # Common HFD layout: org__name
    return repo_id.replace("/", "__")


def list_orphan_dirs(root: Path, referenced: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    keep_names = {_local_name_for_repo(repo) for repo in referenced}
    keep_names.update(referenced)  # also allow org/name as path segments
    orphans: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("."):
            continue
        if name in keep_names:
            continue
        # HF hub cache layout models--org--name
        hub_style = name.startswith("models--") and name.replace("models--", "").replace("--", "/") in referenced
        if hub_style:
            continue
        orphans.append(child)
    return orphans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Checkpoint root (default: resolve_hfd_root(), fallback resolve_ckpt_dir())",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=f"Catalog directory (default: {DEFAULT_MANIFEST_DIR})",
    )
    parser.add_argument("--delete", action="store_true", help="Delete orphan directories (default: dry-run)")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    root = args.root
    if root is None:
        root = resolve_hfd_root()
        if not root.is_dir():
            root = resolve_ckpt_dir()
    root = root.expanduser()
    refs = referenced_repo_ids(args.manifest_dir)
    orphans = list_orphan_dirs(root, refs)
    report = {
        "root": str(root),
        "referenced_repos": sorted(refs),
        "orphans": [str(path) for path in orphans],
        "deleted": [],
        "dry_run": not args.delete,
    }
    if args.delete:
        deleted: list[str] = []
        for path in orphans:
            shutil.rmtree(path)
            deleted.append(str(path))
        report["deleted"] = deleted

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"checkpoint root: {root}")
        print(f"catalog refs: {len(refs)}")
        print(f"orphans ({len(orphans)}){' [dry-run]' if not args.delete else ''}:")
        for path in orphans:
            print(f"  - {path}")
        if args.delete:
            print(f"deleted: {len(report['deleted'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
