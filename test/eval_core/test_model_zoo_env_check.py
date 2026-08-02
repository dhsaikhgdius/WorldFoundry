from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / "model_zoo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_env_check_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(manifest_dir: Path, payload: object) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "models.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_ready_checkpoint(cache_dir: Path, repo_id: str) -> None:
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_text("weights", encoding="utf-8")


def _patch_ready_tools(env_check: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_check.sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(
        env_check,
        "check_command",
        lambda name: {"ok": True, "name": name, "path": f"/usr/bin/{name}"},
    )


def test_checkpoint_repo_filter_only_checks_selected_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_check = _load_script("env_check")
    _patch_ready_tools(env_check, monkeypatch)
    manifest = env_check.verify_sources.ModelManifest(
        model_id="multi",
        path=tmp_path / "multi.json",
        data={"checkpoint": {"repos": [{"id": "org/unselected"}, {"id": "org/selected"}]}},
    )
    cache_dir = tmp_path / "cache" / "hfd"
    _write_ready_checkpoint(cache_dir, "org/selected")

    result = env_check.check_manifest(
        manifest,
        clone_root=tmp_path / "repos",
        cache_dir=cache_dir,
        require_repo=False,
        require_checkpoint=True,
        require_demo=False,
        require_runner_demo=False,
        checkpoint_repo_ids=["org/selected"],
    )

    assert result["ok"] is True
    assert result["available_checkpoint_repo_ids"] == ["org/unselected", "org/selected"]
    assert result["selected_checkpoint_repo_ids"] == ["org/selected"]
    assert [check["repo_id"] for check in result["checkpoint_checks"]] == ["org/selected"]


def test_checkpoint_repo_filter_without_match_fails_cli_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_check = _load_script("env_check")
    _patch_ready_tools(env_check, monkeypatch)
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir,
        {
            "models": [
                {
                    "model_id": "multi",
                    "checkpoint": {"repos": [{"id": "org/a"}, {"id": "org/b"}]},
                }
            ]
        },
    )

    exit_code = env_check.main(
        [
            "--manifest-dir",
            str(manifest_dir),
            "--model-id",
            "multi",
            "--checkpoint-repo-id",
            "org/missing",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "multi: failed" in captured.out
    assert "none of the requested repo ids are listed for this model: org/missing" in captured.out
