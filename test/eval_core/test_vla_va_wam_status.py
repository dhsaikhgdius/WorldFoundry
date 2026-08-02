from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "vla_va_wam_status.py"
    spec = importlib.util.spec_from_file_location("test_vla_va_wam_status_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, targets: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"targets": targets}), encoding="utf-8")
    return path


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _make_git_repo(path: Path, origin: str) -> str:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.email", "status-test@example.invalid")
    _git(path, "config", "user.name", "Status Test")
    (path / "README.md").write_text("ok\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    _git(path, "remote", "add", "origin", origin)
    return _git(path, "rev-parse", "HEAD")


def test_matched_repo_reports_head_remote_and_revision(tmp_path: Path) -> None:
    status = _load_script()
    origin = "https://github.com/example/project"
    repo_path = tmp_path / "cache" / "repos" / "example--project"
    head = _make_git_repo(repo_path, origin)
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [
            {
                "id": "alpha",
                "official_sources": {
                    "github": [{"url": origin, "revision": head, "clone": True}],
                },
            }
        ],
    )

    report = status.build_report(manifest, tmp_path / "cache", ["alpha"])

    repo = report["github_repos"][0]
    assert repo["status"] == "matched"
    assert repo["matched"] is True
    assert repo["expected_revision"] == head
    assert repo["actual_head"] == head
    assert repo["remote_url"] == origin
    assert repo["local_path"] == str(repo_path)


def test_pending_hf_download_reports_metadata_and_pending_counts(tmp_path: Path, monkeypatch) -> None:
    status = _load_script()
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_appear")
    model_path = tmp_path / "cache" / "hf_models" / "org--model"
    (model_path / ".hfd").mkdir(parents=True)
    (model_path / ".hfd" / "repo_metadata.json").write_text(
        json.dumps(
            {
                "sha": "abc123",
                "gated": False,
                "cardData": {"license": "apache-2.0"},
            }
        ),
        encoding="utf-8",
    )
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "weights.safetensors.aria2").write_text("pending", encoding="utf-8")
    (model_path / "chunk.bin.incomplete").write_text("partial", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [
            {
                "id": "beta",
                "official_sources": {
                    "huggingface_models": [
                        {"repo_id": "org/model", "revision": "abc123", "license": "mit"}
                    ],
                },
            }
        ],
    )

    report = status.build_report(manifest, tmp_path / "cache", ["beta"])
    report_text = json.dumps(report, sort_keys=True)

    model = report["hf_models"][0]
    assert model["status"] == "pending"
    assert model["metadata_sha"] == "abc123"
    assert model["metadata_license"] == "apache-2.0"
    assert model["metadata_gated"] is False
    assert model["file_count"] == 1
    assert model["total_bytes"] == 2
    assert model["pending_aria2_count"] == 1
    assert model["pending_incomplete_count"] == 1
    assert "hf_should_not_appear" not in report_text


def test_main_writes_markdown_output(tmp_path: Path) -> None:
    status = _load_script()
    dataset_path = tmp_path / "cache" / "hf_datasets" / "org--dataset"
    (dataset_path / ".hfd").mkdir(parents=True)
    (dataset_path / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "def456", "gated": True, "cardData": {"license": "cc-by-4.0"}}),
        encoding="utf-8",
    )
    (dataset_path / "data.jsonl").write_text("row\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [
            {
                "id": "gamma",
                "official_sources": {
                    "huggingface_datasets": [
                        {"repo_id": "org/dataset", "revision": "def456", "license": "unknown"}
                    ],
                },
            }
        ],
    )
    output_md = tmp_path / "status.md"
    output_json = tmp_path / "status.json"

    rc = status.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path / "cache"),
            "--target",
            "gamma",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    markdown = output_md.read_text(encoding="utf-8")
    assert "# VLA / VA / WAM Acquisition Status" in markdown
    assert "## Hugging Face Datasets" in markdown
    assert "org/dataset" in markdown
    assert "cc-by-4.0" in markdown
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["hf_datasets"][0]["repo_id"] == "org/dataset"
