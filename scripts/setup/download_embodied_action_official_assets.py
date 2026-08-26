#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from worldfoundry.core.io.paths import project_root
from worldfoundry.core.process import run_logged_subprocess

REPO_ROOT = project_root(__file__)
DEFAULT_HF_ROOT = Path(os.environ.get("WORLDFOUNDRY_HFD_ROOT", REPO_ROOT / "cache" / "checkpoints" / "hfd"))
DEFAULT_ASSET_ROOT = Path(os.environ.get("WORLDFOUNDRY_ASSET_ROOT", REPO_ROOT / "cache" / "assets"))
DEFAULT_OPENPI_ROOT = Path(os.environ.get("WORLDFOUNDRY_OPENPI_ASSET_ROOT", DEFAULT_ASSET_ROOT / "openpi"))
DEFAULT_REPOS_ROOT = Path(os.environ.get("WORLDFOUNDRY_MODEL_SOURCE_DIR", REPO_ROOT / "cache" / "models" / "sources"))


@dataclass(frozen=True)
class Asset:
    model_id: str
    kind: str
    asset_id: str
    local_path: str
    revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "asset_id": self.asset_id,
            "local_path": self.local_path,
            "revision": self.revision,
            "metadata": self.metadata,
        }


def _repo_local_dir(root: Path, repo_id: str) -> Path:
    return root / repo_id.replace("/", "--")


def default_assets(
    hf_root: Path = DEFAULT_HF_ROOT,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    openpi_root: Path = DEFAULT_OPENPI_ROOT,
    repos_root: Path = DEFAULT_REPOS_ROOT,
) -> list[Asset]:
    del repos_root
    return [
        Asset(
            "giga-brain-0",
            "hf_model",
            "open-gigaai/GigaBrain-0-3.5B-Base",
            str(_repo_local_dir(hf_root, "open-gigaai/GigaBrain-0-3.5B-Base")),
            metadata={"demo": "GigaBrain-0 official 3.5B base checkpoint"},
        ),
        Asset(
            "giga-brain-0",
            "hf_model",
            "open-gigaai/GigaBrain-0.1-3.5B-Base",
            str(_repo_local_dir(hf_root, "open-gigaai/GigaBrain-0.1-3.5B-Base")),
            metadata={"demo": "GigaBrain-0.1 official 3.5B base checkpoint"},
        ),
        Asset(
            "giga-brain-0",
            "hf_model",
            "google/paligemma-3b-pt-224",
            str(_repo_local_dir(hf_root, "google/paligemma-3b-pt-224")),
            metadata={"access": "manual_terms"},
        ),
        Asset(
            "giga-brain-0",
            "hf_model",
            "physical-intelligence/fast",
            str(_repo_local_dir(hf_root, "physical-intelligence/fast")),
            metadata={"role": "FAST tokenizer"},
        ),
        Asset(
            "gr00t",
            "hf_model",
            "nvidia/GR00T-N1.7-LIBERO",
            str(_repo_local_dir(hf_root, "nvidia/GR00T-N1.7-LIBERO")),
            metadata={"access": "gated_nvidia_terms"},
        ),
        Asset(
            "gr00t",
            "hf_model",
            "nvidia/Cosmos-Reason2-2B",
            str(_repo_local_dir(hf_root, "nvidia/Cosmos-Reason2-2B")),
            revision="9ce19a195e423419c349abfc86fd07178b230561",
            metadata={"access": "gated_nvidia_terms"},
        ),
        Asset(
            "lingbot-va",
            "hf_model",
            "robbyant/lingbot-va-base",
            str(_repo_local_dir(hf_root, "robbyant/lingbot-va-base")),
            metadata={"access": "hf_terms_if_gated"},
        ),
        Asset(
            "lingbot-va",
            "hf_model",
            "robbyant/lingbot-va-posttrain-robotwin",
            str(_repo_local_dir(hf_root, "robbyant/lingbot-va-posttrain-robotwin")),
            metadata={"access": "hf_terms_if_gated"},
        ),
        Asset(
            "openpi",
            "gcs_reference",
            "gs://openpi-assets/checkpoints/pi05_libero",
            str(openpi_root / "checkpoints" / "pi05_libero"),
            metadata={"access": "manual_gcs_or_existing"},
        ),
        Asset(
            "openvla",
            "hf_model",
            "openvla/openvla-7b",
            str(_repo_local_dir(hf_root, "openvla/openvla-7b")),
            metadata={"access": "hf_terms_if_gated"},
        ),
    ]


def _metadata_siblings(local_dir: Path) -> list[str]:
    metadata_path = local_dir / ".hfd" / "repo_metadata.json"
    if not metadata_path.is_file():
        return []
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    siblings = payload.get("siblings") or []
    names: list[str] = []
    for item in siblings:
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str):
            names.append(item["rfilename"])
    return names


def _asset_ready(asset: Asset) -> bool:
    local_dir = Path(asset.local_path)
    if asset.kind == "gcs_reference":
        return local_dir.exists() and any(path.is_file() for path in local_dir.rglob("*"))
    if not local_dir.exists():
        return False
    required = [
        name
        for name in _metadata_siblings(local_dir)
        if Path(name).name.lower() not in {"readme.md", ".gitattributes"} and not name.endswith("/")
    ]
    if required:
        return all((local_dir / name).is_file() for name in required)
    return any(
        path.is_file()
        and ".hfd" not in path.relative_to(local_dir).parts
        and ".git" not in path.relative_to(local_dir).parts
        and path.name.lower() not in {"readme.md", ".gitattributes"}
        for path in local_dir.rglob("*")
    )


def _hf_command(asset: Asset, args: argparse.Namespace) -> list[str]:
    hf = shutil.which("hf") or shutil.which("huggingface-cli") or "hf"
    cmd = [hf, "download", asset.asset_id, "--local-dir", asset.local_path]
    if asset.revision:
        cmd.extend(["--revision", asset.revision])
    if args.max_workers:
        cmd.extend(["--max-workers", str(args.max_workers)])
    return cmd


def _download_hf(asset: Asset, args: argparse.Namespace, env: dict[str, str], log_dir: Path) -> dict[str, Any]:
    path_ready = _asset_ready(asset)
    command = _hf_command(asset, args)
    log_path = log_dir / asset.model_id / f"{asset.asset_id.replace('/', '--')}.log"
    event = {
        "schema_version": "worldfoundry-embodied-action-asset-download",
        **asset.to_dict(),
        "command": command,
        "path_ready": path_ready,
        "log_path": str(log_path),
    }
    if args.list:
        event["status"] = "listed"
        return event
    if args.skip_existing and path_ready:
        event["status"] = "ready"
        return event
    if args.plan_only or asset.kind != "hf_model":
        event["status"] = "planned" if asset.kind == "hf_model" else "manual"
        return event

    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path.with_suffix(log_path.suffix + ".stdout")
    stderr_path = log_path.with_suffix(log_path.suffix + ".stderr")
    start = time.monotonic()
    completed = run_logged_subprocess(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        env=env,
        timeout=args.url_timeout_seconds,
    )
    out = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    err = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    log_path.write_text(out + err, encoding="utf-8")
    event.update(
        {
            "status": "ready" if completed.returncode == 0 and _asset_ready(asset) else "failed",
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - start, 3),
            "path_ready": _asset_ready(asset),
        }
    )
    return event


def _select_assets(assets: Iterable[Asset], models: list[str]) -> list[Asset]:
    requested = {item for item in models if item and item != "all"}
    if not requested:
        return list(assets)
    selected = [asset for asset in assets if asset.model_id in requested or asset.asset_id in requested]
    missing = sorted(requested - {asset.model_id for asset in selected} - {asset.asset_id for asset in selected})
    if missing:
        raise ValueError(f"unknown embodied asset selection: {', '.join(missing)}")
    return selected


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download or check official embodied action model assets.")
    parser.add_argument("models", nargs="*", help="Model ids, asset ids, or all.")
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--repos-root", type=Path, default=DEFAULT_REPOS_ROOT)
    parser.add_argument("--report-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--url-timeout-seconds", type=int, default=3600)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true", help="Write asset commands and local-readiness checks without downloading.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        assets = _select_assets(default_assets(args.hf_root, args.asset_root, args.openpi_root, args.repos_root), args.models)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log_dir = args.log_dir or args.asset_root / "logs" / "embodied_action_assets"
    env = os.environ.copy()
    rows: list[dict[str, Any]] = []
    if args.plan_only or args.list or len(assets) <= 1:
        rows = [_download_hf(asset, args, env, log_dir) for asset in assets]
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers or 1)) as pool:
            futures = [pool.submit(_download_hf, asset, args, env, log_dir) for asset in assets]
            rows = [future.result() for future in as_completed(futures)]
        rows.sort(key=lambda item: (str(item.get("model_id")), str(item.get("asset_id"))))

    summary = {
        "schema_version": "worldfoundry-embodied-action-assets-summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "list" if args.list else ("plan" if args.plan_only else "execute"),
        "asset_count": len(rows),
        "ready_count": sum(1 for row in rows if row.get("status") == "ready"),
        "failed_count": sum(1 for row in rows if row.get("status") == "failed"),
        "planned_count": sum(1 for row in rows if row.get("status") == "planned"),
        "manual_count": sum(1 for row in rows if row.get("status") == "manual"),
        "ok": all(row.get("status") != "failed" for row in rows),
    }
    if args.report_jsonl:
        _write_jsonl(args.report_jsonl, rows)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({"summary": summary, "assets": rows}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['model_id']}\t{row['asset_id']}\t{row['status']}\t{row['local_path']}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
