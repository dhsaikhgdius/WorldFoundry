"""R-06: record per-file checkpoint sizes and verify them on local checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "model_zoo" / "download_checkpoints.py"
    spec = importlib.util.spec_from_file_location("test_r06_download_checkpoints", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


def _make_direct_repo(cache_dir: Path) -> Path:
    repo_dir = cache_dir / "Example--Model"
    (repo_dir / "context").mkdir(parents=True)
    (repo_dir / "context" / "weights.safetensors").write_bytes(b"x" * 64)
    (repo_dir / "config.json").write_bytes(b"{}")
    return repo_dir


def test_record_then_check_roundtrip_direct_layout(script: ModuleType, tmp_path: Path) -> None:
    _make_direct_repo(tmp_path)

    recorded = script.record_checkpoint_file_sizes("Example/Model", tmp_path)
    assert recorded["written"] is True
    assert recorded["layout"] == "direct_hfd"
    assert recorded["files"] == {"config.json": 2, "context/weights.safetensors": 64}
    manifest_path = Path(recorded["path"])
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == script.SIZE_MANIFEST_SCHEMA

    check = script.check_local_checkpoint("Example/Model", tmp_path)
    assert check["size_manifest_found"] is True
    assert check["size_mismatches"] == []
    assert check["size_check_ok"] is True
    assert check["ready"] is True


def test_truncated_and_deleted_files_fail_the_size_check(script: ModuleType, tmp_path: Path) -> None:
    repo_dir = _make_direct_repo(tmp_path)
    script.record_checkpoint_file_sizes("Example/Model", tmp_path)

    (repo_dir / "context" / "weights.safetensors").write_bytes(b"x" * 10)
    (repo_dir / "config.json").unlink()

    check = script.check_local_checkpoint("Example/Model", tmp_path)
    assert check["size_check_ok"] is False
    assert check["ready"] is False
    mismatches = {item["path"]: item for item in check["size_mismatches"]}
    assert mismatches["context/weights.safetensors"]["expected_size_bytes"] == 64
    assert mismatches["context/weights.safetensors"]["actual_size_bytes"] == 10
    assert mismatches["config.json"]["actual_size_bytes"] is None


def test_missing_size_manifest_is_backward_compatible(script: ModuleType, tmp_path: Path) -> None:
    _make_direct_repo(tmp_path)

    check = script.check_local_checkpoint("Example/Model", tmp_path)
    assert check["size_manifest_found"] is False
    assert check["size_check_ok"] is True
    assert check["ready"] is True


def test_record_then_check_hf_cache_layout(script: ModuleType, tmp_path: Path) -> None:
    snapshot = "a" * 40
    repo_dir = tmp_path / "models--Example--Model"
    snapshot_dir = repo_dir / "snapshots" / snapshot
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.bin").write_bytes(b"y" * 32)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(snapshot, encoding="utf-8")

    recorded = script.record_checkpoint_file_sizes("Example/Model", tmp_path)
    assert recorded["layout"] == "hf_cache"
    assert recorded["files"] == {f"{snapshot}/model.bin": 32}

    check = script.check_local_checkpoint("Example/Model", tmp_path)
    assert check["size_check_ok"] is True
    assert check["ready"] is True

    (snapshot_dir / "model.bin").write_bytes(b"y" * 5)
    check = script.check_local_checkpoint("Example/Model", tmp_path)
    assert check["size_check_ok"] is False
    assert check["ready"] is False
    assert check["size_mismatches"] == [
        {
            "path": f"{snapshot}/model.bin",
            "expected_size_bytes": 32,
            "actual_size_bytes": 5,
        }
    ]


def test_execute_download_records_size_manifest(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = script.ModelManifest(
        model_id="example-model",
        path=tmp_path / "example-model.yaml",
        data={"id": "example-model", "hf_repo_id": "Example/Model"},
    )

    def fake_run(command: list[str], env: dict[str, str], **kwargs: object) -> dict[str, object]:
        _make_direct_repo(tmp_path)
        return {"command": command, "ok": True, "returncode": 0, "attempts": []}

    monkeypatch.setattr(script, "find_hf_downloader", lambda: ["hf", "download"])
    monkeypatch.setattr(script, "run_download_command", fake_run)

    result = script.download_manifest(manifest, tmp_path, execute=True, check_local=True)
    assert result["ok"] is True
    assert len(result["size_manifests"]) == 1
    assert result["size_manifests"][0]["written"] is True
    assert result["local_checks"][0]["size_check_ok"] is True
    assert Path(result["size_manifests"][0]["path"]).is_file()
