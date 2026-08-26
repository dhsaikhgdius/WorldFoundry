#!/usr/bin/env python3
"""Fetch pinned Studio Spark/Three vendor JS into the gitignored vendor tree.

PK-01: ``worldfoundry/studio/assets/vendor/`` is gitignored, so a clean clone
404s Spark/Three module URLs unless these files are provisioned. This script
downloads pinned CDN URLs and verifies sha256 digests from
``vendor_assets.manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "vendor_assets.manifest.json"
DEFAULT_VENDOR_DIR = REPO_ROOT / "worldfoundry" / "studio" / "assets" / "vendor"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ValueError(f"invalid vendor manifest: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_asset(url: str, destination: Path, expected_sha256: str, *, force: bool) -> str:
    if destination.is_file() and not force:
        actual = sha256_file(destination)
        if actual == expected_sha256:
            return "ok"
        raise SystemExit(
            f"hash mismatch for existing {destination}: got {actual}, expected {expected_sha256}. "
            "Re-run with --force to replace."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        actual = sha256_file(tmp_path)
        if actual != expected_sha256:
            raise SystemExit(f"downloaded hash mismatch for {url}: got {actual}, expected {expected_sha256}")
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)
    return "fetched"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=DEFAULT_VENDOR_DIR,
        help="Destination vendor root (default: worldfoundry/studio/assets/vendor)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even when hashes already match.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify local files against the manifest; do not download.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest()
    vendor_dir = args.vendor_dir.resolve()
    statuses: list[str] = []
    for asset in manifest["assets"]:
        relative = str(asset["relative_path"])
        destination = vendor_dir / relative
        expected = str(asset["sha256"])
        url = str(asset["url"])
        if args.check:
            if not destination.is_file():
                print(f"missing: {destination}", file=sys.stderr)
                return 1
            actual = sha256_file(destination)
            if actual != expected:
                print(f"hash mismatch: {destination}", file=sys.stderr)
                return 1
            statuses.append(f"ok {relative}")
            continue
        status = fetch_asset(url, destination, expected, force=args.force)
        statuses.append(f"{status} {relative}")

    for line in statuses:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
