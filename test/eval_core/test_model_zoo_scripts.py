from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / "model_zoo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
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


def test_verify_sources_uses_git_ls_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_sources = _load_script("verify_sources")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="abc\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(verify_sources.subprocess, "run", fake_run)

    result = verify_sources.check_git_repo("https://github.com/example/project.git", 5)

    assert result["ok"] is True
    assert calls == [["git", "ls-remote", "https://github.com/example/project.git", "HEAD"]]


def test_verify_sources_normalizes_github_repo_url_for_git_ls_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_sources = _load_script("verify_sources")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="abc\tHEAD\n", stderr="")

    monkeypatch.setattr(verify_sources.subprocess, "run", fake_run)

    result = verify_sources.check_git_repo("https://github.com/example/project", 5)

    assert result["ok"] is True
    assert result["url"] == "https://github.com/example/project.git"
    assert calls == [["git", "ls-remote", "https://github.com/example/project.git", "HEAD"]]


def test_verify_sources_checks_huggingface_api_with_head(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_sources = _load_script("verify_sources")
    requests: list[urllib.request.Request] = []

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
        requests.append(request)
        assert timeout == 7
        return FakeResponse()

    monkeypatch.setattr(verify_sources.urllib.request, "urlopen", fake_urlopen)

    result = verify_sources.check_hf_repo("org/model-name", 7)

    assert result["ok"] is True
    assert requests[0].get_method() == "HEAD"
    assert requests[0].full_url == "https://huggingface.co/api/models/org/model-name"


def test_verify_sources_reports_git_timeout_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_sources = _load_script("verify_sources")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=3)

    monkeypatch.setattr(verify_sources.subprocess, "run", fake_run)

    result = verify_sources.check_git_repo("https://github.com/example/project.git", 3)

    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_verify_sources_loads_entries_and_all_nested_hf_repos(tmp_path: Path) -> None:
    verify_sources = _load_script("verify_sources")
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir,
        {
            "entries": [
                {
                    "id": "multi-repo",
                    "source_status": {"github": {"url": "https://github.com/example/multi"}},
                    "checkpoint_refs": [
                        {"repo_id": "org/model-a"},
                        {"repo_id": "org/model-b"},
                    ],
                }
            ]
        },
    )

    manifests = verify_sources.load_manifests(manifest_dir, "multi-repo")

    assert len(manifests) == 1
    assert manifests[0].official_repo_urls == ["https://github.com/example/multi"]
    assert manifests[0].hf_repo_ids == ["org/model-a", "org/model-b"]


def test_download_checkpoints_plan_only_filters_model_and_never_calls_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir,
        {
            "models": [
                {"model_id": "alpha", "checkpoint_refs": [{"repo_id": "org/alpha"}]},
                {"model_id": "beta", "checkpoint_refs": [{"repo_id": "org/beta"}]},
            ]
        },
    )

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("plan-only must not execute a download command")

    monkeypatch.setattr(download_checkpoints.subprocess, "run", fail_run)

    manifests = download_checkpoints.load_manifests(manifest_dir, model_id="beta")
    results = [
        download_checkpoints.download_manifest(manifest, tmp_path / "cache" / "hfd", execute=False)
        for manifest in manifests
    ]

    assert [result["model_id"] for result in results] == ["beta"]
    assert results[0]["executed"] is False
    assert results[0]["command"] == [
        "hf",
        "download",
        "org/beta",
        "--cache-dir",
        str(tmp_path / "cache" / "hfd"),
        "--max-workers",
        "1",
    ]


def test_download_checkpoints_loads_nested_catalog_manifests(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir / "world_models",
        {
            "model_id": "matrix-game-2",
            "checkpoint": {
                "repos": [
                    {
                        "id": "Skywork/Matrix-Game-2.0",
                        "sha": "f1729d99a80e0f07993a77d7dad4a3190e23c2c8",
                    }
                ]
            },
        },
    )

    manifests = download_checkpoints.load_manifests(manifest_dir, model_id="matrix-game-2")

    assert len(manifests) == 1
    assert manifests[0].hf_repo_ids == ["Skywork/Matrix-Game-2.0"]
    assert manifests[0].hf_repo_revisions == {
        "Skywork/Matrix-Game-2.0": "f1729d99a80e0f07993a77d7dad4a3190e23c2c8"
    }


def test_download_checkpoints_execute_calls_selected_hf_downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="alpha",
        path=tmp_path / "alpha.yaml",
        data={"checkpoint_refs": [{"repo_id": "org/alpha"}]},
    )
    calls: list[list[str]] = []

    def fake_build_download_command(
        repo_id: str,
        cache_dir: Path,
        revision: str | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        assert revision is None
        assert max_workers == 1
        return ["huggingface-cli", "download", repo_id, "--cache-dir", str(cache_dir)]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(download_checkpoints, "build_download_command", fake_build_download_command)
    monkeypatch.setattr(download_checkpoints.subprocess, "run", fake_run)

    result = download_checkpoints.download_manifest(manifest, tmp_path / "cache" / "hfd", execute=True)

    assert result["ok"] is True
    assert calls == [["huggingface-cli", "download", "org/alpha", "--cache-dir", str(tmp_path / "cache" / "hfd")]]


def test_download_checkpoints_repo_id_filter_limits_multi_checkpoint_manifest(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="multi",
        path=tmp_path / "multi.yaml",
        data={"checkpoint_refs": [{"repo_id": "org/a"}, {"repo_id": "org/b"}]},
    )

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        repo_id_filter=["org/b"],
    )

    assert result["ok"] is True
    assert result["hf_repo_ids"] == ["org/b"]
    assert result["commands"] == [
        ["hf", "download", "org/b", "--cache-dir", str(tmp_path / "cache" / "hfd"), "--max-workers", "1"]
    ]


def test_download_checkpoints_plan_only_includes_manifest_revision(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="revisioned",
        path=tmp_path / "revisioned.yaml",
        data={"checkpoint_refs": [{"repo_id": "org/model", "sha": "abc1234"}]},
    )

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["commands"] == [
        [
            "hf",
            "download",
            "org/model",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "abc1234",
            "--max-workers",
            "1",
        ]
    ]


def test_download_checkpoints_reads_variant_checkpoint_refs(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="variant-model",
        path=tmp_path / "variant-model.yaml",
        data={
            "variants": [
                {
                    "id": "small",
                    "checkpoint_refs": [{"repo_id": "org/model-small", "revision": "abc1234"}],
                },
                {
                    "id": "large",
                    "checkpoint_refs": [{"repo_id": "org/model-large", "sha": "def5678"}],
                },
            ]
        },
    )

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["hf_repo_ids"] == ["org/model-small", "org/model-large"]
    assert result["commands"] == [
        [
            "hf",
            "download",
            "org/model-small",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "abc1234",
            "--max-workers",
            "1",
        ],
        [
            "hf",
            "download",
            "org/model-large",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "def5678",
            "--max-workers",
            "1",
        ],
    ]


def test_download_checkpoints_execute_retries_and_preserves_proxy_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="alpha",
        path=tmp_path / "alpha.json",
        data={"hf_repo_id": "org/alpha"},
    )
    calls: list[tuple[list[str], dict[str, str], int | None]] = []

    def fake_find_hf_downloader() -> list[str]:
        return ["hf", "download"]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        calls.append((command, env, kwargs.get("timeout")))
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(download_checkpoints, "find_hf_downloader", fake_find_hf_downloader)
    monkeypatch.setattr(download_checkpoints.subprocess, "run", fake_run)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=True,
        check_local=False,
        env_overrides={"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"},
        timeout_seconds=123,
        retries=1,
        max_workers=1,
    )

    assert result["ok"] is True
    assert len(calls) == 2
    assert calls[0][0] == [
        "hf",
        "download",
        "org/alpha",
        "--cache-dir",
        str(tmp_path / "cache" / "hfd"),
        "--max-workers",
        "1",
    ]
    assert calls[0][1]["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert calls[0][1]["HF_HUB_DISABLE_XET"] == "1"
    assert calls[0][1]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert calls[0][2] == 123
    assert result["download_runs"][0]["attempt_count"] == 2
    assert result["download_options"]["env"]["proxy_env_keys"]


def test_download_checkpoints_execute_records_timeout_without_deleting_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    incomplete = cache_dir / "models--org--alpha" / "blobs" / "part.incomplete"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text("partial", encoding="utf-8")
    manifest = download_checkpoints.ModelManifest(
        model_id="alpha",
        path=tmp_path / "alpha.json",
        data={"hf_repo_id": "org/alpha"},
    )

    def fake_find_hf_downloader() -> list[str]:
        return ["hf", "download"]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(download_checkpoints, "find_hf_downloader", fake_find_hf_downloader)
    monkeypatch.setattr(download_checkpoints.subprocess, "run", fake_run)

    result = download_checkpoints.download_manifest(
        manifest,
        cache_dir,
        execute=True,
        check_local=True,
        timeout_seconds=5,
        retries=0,
    )

    assert result["ok"] is False
    assert result["download_runs"][0]["attempts"][0]["timed_out"] is True
    assert result["local_checks"][0]["incomplete_files"][0]["path"].endswith("part.incomplete")
    assert incomplete.exists()


def test_download_checkpoints_main_writes_structured_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest_dir = tmp_path / "model_zoo"
    report_path = tmp_path / "report.json"
    _write_manifest(manifest_dir, {"model_id": "alpha", "hf_repo_id": "org/alpha"})

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("check mode must not execute a download command")

    monkeypatch.setattr(download_checkpoints.subprocess, "run", fail_run)

    exit_code = download_checkpoints.main(
        [
            "--manifest-dir",
            str(manifest_dir),
            "--model-id",
            "alpha",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--report-path",
            str(report_path),
            "--json",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "worldfoundry-model-zoo-checkpoint-download-report"
    assert report["mode"] == "check"
    assert report["check_only"] is True
    assert report["ok"] is True
    assert report["results"][0]["download_options"]["max_workers"] == 1


def test_download_checkpoints_local_check_detects_ready_snapshot(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    blob = repo_dir / "blobs" / "blob-a"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    blob.write_text("weights", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/blob-a")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir)

    assert result["ready"] is True
    assert result["file_count"] == 1
    assert result["incomplete_files"] == []


def test_download_checkpoints_local_check_accepts_symlinked_hf_repo_root(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    shared_repo_dir = tmp_path / "shared" / "models--org--model"
    revision = "abc123def456"
    snapshot = shared_repo_dir / "snapshots" / revision
    blob = shared_repo_dir / "blobs" / "blob-a"
    (shared_repo_dir / "refs").mkdir(parents=True)
    (shared_repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (shared_repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    blob.write_text("weights", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/blob-a")
    cache_dir.mkdir(parents=True)
    (cache_dir / "models--org--model").symlink_to(shared_repo_dir, target_is_directory=True)

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision=revision)

    assert result["ready"] is True
    assert result["local_layout"] == "hf_cache"
    assert result["file_count"] == 1
    assert result["broken_links"] == []


def test_download_checkpoints_local_check_detects_incomplete_file(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (repo_dir / "blobs" / "blob-a.incomplete").write_text("partial", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir)

    assert result["ready"] is False
    assert result["incomplete_files"][0]["path"].endswith("blob-a.incomplete")
    assert result["blocking_incomplete_files"][0]["path"].endswith("blob-a.incomplete")


def test_download_checkpoints_local_check_allows_orphan_incomplete_file(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    blob = repo_dir / "blobs" / "blob-a"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    blob.write_text("weights", encoding="utf-8")
    (repo_dir / "blobs" / "orphan.incomplete").write_text("partial", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/blob-a")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir)

    assert result["ready"] is True
    assert result["blocking_incomplete_files"] == []
    assert result["orphan_incomplete_files"][0]["path"].endswith("orphan.incomplete")


def test_download_checkpoints_local_check_requires_expected_revision(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    (repo_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("weights", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="def456")

    assert result["ready"] is False
    assert result["expected_revision"] == "def456"
    assert result["revision_matches"] is False


def test_download_checkpoints_local_check_accepts_direct_hfd_layout(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "hfd"
    repo_dir = cache_dir / "org--model"
    (repo_dir / ".hfd").mkdir(parents=True)
    (repo_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456"}),
        encoding="utf-8",
    )
    (repo_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is True
    assert result["local_layout"] == "direct_hfd"
    assert result["direct_hfd_file_count"] == 1
    assert result["direct_hfd_revision"] == "abc123def456"
    assert result["direct_hfd_revision_matches"] is True


def test_download_checkpoints_local_check_accepts_snapshot_download_local_dir(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "hfd"
    repo_dir = cache_dir / "org--model"
    metadata_dir = repo_dir / ".cache" / "huggingface" / "download" / "subdir"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "model.safetensors.metadata").write_text(
        "abc123def456\netag\n123.0\n",
        encoding="utf-8",
    )
    (repo_dir / "subdir").mkdir()
    (repo_dir / "subdir" / "model.safetensors").write_text("weights", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is True
    assert result["local_layout"] == "direct_hfd"
    assert result["direct_hfd_file_count"] == 1
    assert result["direct_hfd_revision"] == "abc123def456"


def test_download_checkpoints_local_check_rejects_direct_hfd_metadata_only(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "hfd"
    repo_dir = cache_dir / "org--model"
    (repo_dir / ".hfd").mkdir(parents=True)
    (repo_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456"}),
        encoding="utf-8",
    )

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is False
    assert result["local_layout"] == "missing"
    assert result["direct_hfd_file_count"] == 0


def test_run_demo_parity_runs_one_manifest_demo_command_and_writes_report(tmp_path: Path) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    manifest = run_demo_parity.ModelManifest(
        model_id="demo-model",
        path=tmp_path / "demo.json",
        data={"demo_command": [sys.executable, "-c", "print('demo-ok')"]},
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    report_path = tmp_path / "parity" / "demo-model" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert report["schema_version"] == "worldfoundry-model-demo-parity-report"
    assert isinstance(report["generated_at"], str)
    assert report["validator"]["script"] == "scripts/model_zoo/run_demo_parity.py"
    assert "git" in report["validator"]
    assert report["model_id"] == "demo-model"
    assert report["returncode"] == 0
    assert report["stdout"] == "demo-ok\n"


def test_run_demo_parity_timeout_decodes_bytes_before_writing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    manifest = run_demo_parity.ModelManifest(
        model_id="timeout-model",
        path=tmp_path / "demo.json",
        data={"demo_command": [sys.executable, "-c", "print('slow')"]},
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=10, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(run_demo_parity.subprocess, "run", fake_run)

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)
    report = json.loads((tmp_path / "parity" / "timeout-model" / "report.json").read_text(encoding="utf-8"))

    assert result["ok"] is False
    assert report["stdout"] == "partial stdout"
    assert report["stderr"] == "partial stderr"
    json.dumps(report)


def test_run_demo_parity_resolves_python_to_current_executable(tmp_path: Path) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    manifest = run_demo_parity.ModelManifest(
        model_id="python-resolution",
        path=tmp_path / "demo.json",
        data={"demo_command": ["python", "-c", "import sys; print(sys.executable)"]},
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    assert result["ok"] is True
    assert result["demo_command"][0] == sys.executable
    assert result["stdout"].strip() == sys.executable


def test_run_demo_parity_respects_python_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    override = sys.executable
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_DEMO_PYTHON", override)
    manifest = run_demo_parity.ModelManifest(
        model_id="python-override",
        path=tmp_path / "demo.json",
        data={"demo_command": ["python", "-c", "import sys; print(sys.executable)"]},
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    assert result["ok"] is True
    assert result["demo_command"][0] == override
    assert result["stdout"].strip() == override


def test_run_demo_parity_loads_entries_and_nested_demo_command(tmp_path: Path) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir,
        {
            "entries": [
                {
                    "id": "nested-demo",
                    "demo_parity": {
                        "demo_command": [sys.executable, "-c", "print('nested-demo-ok')"],
                    },
                }
            ]
        },
    )

    manifests = run_demo_parity.load_manifests(manifest_dir, "nested-demo")
    result = run_demo_parity.run_demo(manifests[0], tmp_path / "parity", timeout_seconds=10)

    assert result["ok"] is True
    assert result["stdout"] == "nested-demo-ok\n"


def test_run_demo_parity_checks_expected_artifacts(tmp_path: Path) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    artifact = tmp_path / "artifact.mp4"
    manifest = run_demo_parity.ModelManifest(
        model_id="artifact-demo",
        path=tmp_path / "demo.json",
        data={
            "demo_command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(artifact)!r}).write_text('ok')",
            ],
            "demo_parity": {"expected_artifacts": [str(artifact)]},
        },
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    assert result["ok"] is True
    assert result["artifact_checks"][0]["exists"] is True
    assert result["artifact_checks"][0]["size_bytes"] == 2


def test_run_demo_parity_checks_expected_artifact_sha256(tmp_path: Path) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    artifact = tmp_path / "artifact.mp4"
    digest = hashlib.sha256(b"ok").hexdigest()
    manifest = run_demo_parity.ModelManifest(
        model_id="artifact-demo",
        path=tmp_path / "demo.json",
        data={
            "demo_command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(artifact)!r}).write_bytes(b'ok')",
            ],
            "demo_parity": {"expected_artifacts": [{"path": str(artifact), "sha256": digest}]},
        },
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    assert result["ok"] is True
    assert result["artifact_checks"][0]["checksum_ok"] is True
    assert result["artifact_checks"][0]["actual_sha256"] == digest


def test_run_demo_parity_checks_video_probe_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    artifact = tmp_path / "artifact.mp4"

    def fake_probe(path: Path) -> dict[str, object]:
        assert path == artifact
        return {"ok": True, "frame_count": 12, "width": 640, "height": 352, "fps": 12.0}

    monkeypatch.setattr(run_demo_parity, "probe_video_artifact", fake_probe)
    manifest = run_demo_parity.ModelManifest(
        model_id="video-probe-demo",
        path=tmp_path / "demo.json",
        data={
            "demo_command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(artifact)!r}).write_bytes(b'not-a-real-video')",
            ],
            "demo_parity": {
                "expected_artifacts": [
                    {
                        "path": str(artifact),
                        "video_probe": {"min_frames": 10, "min_width": 600, "min_height": 300, "min_fps": 10},
                    }
                ]
            },
        },
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    assert result["ok"] is True
    video_probe = result["artifact_checks"][0]["video_probe"]
    assert video_probe["ok"] is True
    assert video_probe["checks"] == {
        "min_frames": True,
        "min_width": True,
        "min_height": True,
        "min_fps": True,
    }


def test_run_demo_parity_fails_video_probe_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    artifact = tmp_path / "artifact.mp4"
    monkeypatch.setattr(
        run_demo_parity,
        "probe_video_artifact",
        lambda path: {"ok": True, "frame_count": 3, "width": 320, "height": 180, "fps": 12.0},
    )
    manifest = run_demo_parity.ModelManifest(
        model_id="video-probe-fail-demo",
        path=tmp_path / "demo.json",
        data={
            "demo_command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(artifact)!r}).write_bytes(b'not-a-real-video')",
            ],
            "demo_parity": {
                "expected_artifacts": [
                    {"path": str(artifact), "video_probe": {"min_frames": 10, "min_width": 600}}
                ]
            },
        },
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    assert result["returncode"] == 0
    assert result["ok"] is False
    assert result["artifact_checks"][0]["video_ok"] is False
    assert result["artifact_checks"][0]["video_probe"]["checks"] == {"min_frames": False, "min_width": False}


def test_run_demo_parity_fails_when_expected_artifact_is_missing(tmp_path: Path) -> None:
    run_demo_parity = _load_script("run_demo_parity")
    manifest = run_demo_parity.ModelManifest(
        model_id="missing-artifact-demo",
        path=tmp_path / "demo.json",
        data={
            "demo_command": [sys.executable, "-c", "print('no artifact')"],
            "demo_parity": {"expected_artifacts": ["missing.mp4"]},
        },
    )

    result = run_demo_parity.run_demo(manifest, tmp_path / "parity", timeout_seconds=10)

    assert result["returncode"] == 0
    assert result["ok"] is False
    assert result["artifact_checks"][0]["exists"] is False


def test_validate_integration_blocks_demo_without_expected_artifacts(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.ModelManifest(
        model_id="no-artifact-contract",
        path=tmp_path / "model.json",
        data={"demo_command": [sys.executable, "-c", "print('ok')"]},
    )

    result = validate_integration.demo_stage(
        manifest,
        tmp_path / "validation",
        execute=False,
        timeout_seconds=10,
        command_kind="official",
    )

    assert result["status"] == "blocked"
    assert result["error"] == "missing official expected_artifacts"


def test_validate_integration_status_blocks_known_stage_failures() -> None:
    validate_integration = _load_script("validate_integration")

    status = validate_integration.integration_status(
        {
            "source": {"ok": True},
            "repo_clone": {"ok": True, "ready": False},
            "environment": {"ok": True},
            "checkpoint": {"ok": False, "ready": False},
            "demo_parity": {"ok": True, "ready": False},
            "runner_parity": {"ok": True, "ready": False},
        }
    )

    assert status == "blocked"


def test_validate_integration_accepts_project_supported_python_range(monkeypatch: pytest.MonkeyPatch) -> None:
    validate_integration = _load_script("validate_integration")

    monkeypatch.setattr(validate_integration.sys, "version_info", (3, 11, 14))
    monkeypatch.setattr(
        validate_integration,
        "check_command",
        lambda name: {"ok": True, "name": name, "path": f"/usr/bin/{name}"},
    )

    result = validate_integration.check_environment(require_hf=False)

    assert result["ok"] is True
    assert result["checks"]["python_supported"]["ok"] is True
    assert result["checks"]["python_supported"]["required"] == ">=3.10,<3.14"


def test_run_demo_parity_cli_requires_model_id() -> None:
    run_demo_parity = _load_script("run_demo_parity")

    with pytest.raises(SystemExit):
        run_demo_parity.build_parser().parse_args([])


def test_validate_integration_plan_only_stops_at_source_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir,
        {
            "models": [
                {
                    "model_id": "alpha",
                    "official_repo_url": "https://github.com/example/alpha",
                    "hf_repo_id": "org/alpha",
                    "demo_command": [sys.executable, "-c", "print('alpha')"],
                    "expected_artifacts": ["alpha.mp4"],
                    "runner_parity": {
                        "demo_command": [sys.executable, "-c", "print('runner')"],
                        "expected_artifacts": ["alpha_runner.mp4"],
                    },
                }
            ]
        },
    )

    def fake_verify_manifest(manifest: object, timeout_seconds: int) -> dict[str, object]:
        return {"model_id": "alpha", "ok": True, "checks": {}}

    monkeypatch.setattr(validate_integration.verify_sources, "verify_manifest", fake_verify_manifest)
    monkeypatch.setattr(
        validate_integration,
        "check_environment",
        lambda require_hf: {"ok": True, "status": "env_ready", "checks": {}},
    )

    manifests = validate_integration.verify_sources.load_manifests(manifest_dir, "alpha")
    report = validate_integration.validate_manifest(
        manifests[0],
        output_root=tmp_path / "validation",
        clone_root=tmp_path / "repos",
        cache_dir=tmp_path / "cache",
        timeout_seconds=10,
        clone_timeout_seconds=10,
        depth=1,
        execute_clone=False,
        update_clone=False,
        fresh_clone=False,
        execute_download=False,
        check_local_checkpoint=False,
        checkpoint_repo_ids=None,
        disable_xet=False,
        execute_official_demo=False,
        execute_runner_demo=False,
    )

    assert report["status"] == "source_verified"
    assert report["integrated"] is False
    assert report["schema_version"] == "worldfoundry-model-integration-report"
    assert report["manifest_sha256"] == validate_integration.json_sha256(manifests[0].data)
    assert report["manifest_sha256_scope"] == "entry"
    assert report["manifest_file_sha256"] == validate_integration.file_sha256(manifests[0].path)
    assert report["validator"]["script"] == "scripts/model_zoo/validate_integration.py"
    assert report["validator"]["script_sha256"] == validate_integration.file_sha256(
        REPO_ROOT / "scripts" / "model_zoo" / "validate_integration.py"
    )
    assert report["validator"]["python"]["version"] == sys.version.split()[0]
    assert isinstance(report["validator"]["git"]["commit"], str)
    assert report["stages"]["repo_clone"]["checks"][0]["executed"] is False
    assert Path(report["report_path"]).is_file()


def test_validate_integration_checkpoint_stage_plan_only_stays_planned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    args = validate_integration.build_parser().parse_args(["--model-id", "alpha"])
    assert args.check_local_checkpoint is False
    manifest = validate_integration.verify_sources.ModelManifest(
        model_id="alpha",
        path=tmp_path / "model.json",
        data={"hf_repo_id": "org/alpha"},
    )
    calls: list[dict[str, object]] = []

    def fake_download_manifest(manifest: object, cache_dir: Path, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "model_id": "alpha",
            "hf_repo_ids": ["org/alpha"],
            "executed": False,
            "local_checks": [],
            "ok": True,
        }

    monkeypatch.setattr(validate_integration.download_checkpoints, "download_manifest", fake_download_manifest)

    result = validate_integration.checkpoint_stage(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        check_local=False,
        repo_id_filter=None,
        disable_xet=False,
    )

    assert result["ready"] is False
    assert result["status"] == "planned"
    assert calls[0]["execute"] is False
    assert calls[0]["check_local"] is False


def test_validate_integration_checkpoint_stage_check_local_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    args = validate_integration.build_parser().parse_args(["--model-id", "alpha", "--check-local"])
    assert args.check_local_checkpoint is True
    manifest = validate_integration.verify_sources.ModelManifest(
        model_id="alpha",
        path=tmp_path / "model.json",
        data={"hf_repo_id": "org/alpha"},
    )
    calls: list[dict[str, object]] = []

    def fake_download_manifest(manifest: object, cache_dir: Path, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "model_id": "alpha",
            "hf_repo_ids": ["org/alpha"],
            "executed": False,
            "local_checks": [{"repo_id": "org/alpha", "ready": True}],
            "ok": True,
        }

    monkeypatch.setattr(validate_integration.download_checkpoints, "download_manifest", fake_download_manifest)

    result = validate_integration.checkpoint_stage(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        check_local=True,
        repo_id_filter=None,
        disable_xet=False,
    )

    assert result["ready"] is True
    assert result["status"] == "checkpoint_ready"
    assert calls[0]["execute"] is False
    assert calls[0]["check_local"] is True


def test_validate_integration_checkpoint_stage_check_local_blocks_on_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.ModelManifest(
        model_id="alpha",
        path=tmp_path / "model.json",
        data={"hf_repo_id": "org/alpha"},
    )

    def fake_download_manifest(manifest: object, cache_dir: Path, **kwargs: object) -> dict[str, object]:
        return {
            "model_id": "alpha",
            "hf_repo_ids": ["org/alpha"],
            "executed": False,
            "local_checks": [{"repo_id": "org/alpha", "ready": False, "issue": "missing snapshot"}],
            "ok": False,
        }

    monkeypatch.setattr(validate_integration.download_checkpoints, "download_manifest", fake_download_manifest)

    result = validate_integration.checkpoint_stage(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        check_local=True,
        repo_id_filter=None,
        disable_xet=False,
    )

    assert result["ready"] is False
    assert result["status"] == "blocked"


def test_validate_integration_clone_executes_git_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[3:] == ["status", "--porcelain", "--untracked-files=no"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="cloned\n", stderr="")

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/project",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=False,
    )

    assert result["ok"] is True
    assert result["ready"] is True
    assert calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/example/project.git",
            str(tmp_path / "repos" / "github.com_example_project"),
        ]
    ]


def test_validate_integration_clone_timeout_decodes_bytes_for_json_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=10, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/project",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=False,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "partial stderr"
    json.dumps(result)


def test_validate_integration_fresh_clone_removes_existing_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    repo_dir = validate_integration.repo_dir_for_url(tmp_path / "repos", "https://github.com/example/project")
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "index.lock").write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert not (repo_dir / ".git" / "index.lock").exists()
        return subprocess.CompletedProcess(command, 0, stdout="cloned\n", stderr="")

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/project",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=True,
    )

    assert result["ok"] is True
    assert result["ready"] is True
    assert result["removed_existing"] is True
    assert calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/example/project.git",
            str(repo_dir),
        ]
    ]


def test_validate_integration_existing_repo_with_index_lock_is_blocked(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    repo_dir = validate_integration.repo_dir_for_url(tmp_path / "repos", "https://github.com/example/project")
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "index.lock").write_text("", encoding="utf-8")

    result = validate_integration.clone_git_repo(
        "https://github.com/example/project",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "index lock" in result["error"]


def test_validate_integration_existing_repo_with_tracked_changes_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    repo_dir = validate_integration.repo_dir_for_url(tmp_path / "repos", "https://github.com/example/project")
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[3:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[3:] == ["status", "--porcelain", "--untracked-files=no"]:
            return subprocess.CompletedProcess(command, 0, stdout=" D file.py\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/project",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "tracked worktree changes" in result["error"]


def test_validate_integration_clone_manifest_repos_surfaces_blocked_child_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.ModelManifest(
        model_id="alpha",
        path=tmp_path / "model.json",
        data={"official_repo_url": "https://github.com/example/project"},
    )

    def fake_clone_git_repo(url: str, clone_root: Path, **kwargs: object) -> dict[str, object]:
        return {"ok": False, "ready": False, "status": "blocked", "error": "partial clone"}

    monkeypatch.setattr(validate_integration, "clone_git_repo", fake_clone_git_repo)

    result = validate_integration.clone_manifest_repos(
        manifest,
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert result["checks"][0]["error"] == "partial clone"


def test_model_env_check_reports_repo_checkpoint_and_demo_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_check = _load_script("env_check")
    manifest = env_check.verify_sources.ModelManifest(
        model_id="ready-model",
        path=tmp_path / "model.json",
        data={
            "official_repo_url": "https://github.com/example/ready-model",
            "hf_repo_id": "org/model",
            "demo_command": [sys.executable, "-c", "print('ok')"],
            "expected_artifacts": ["demo.mp4"],
        },
    )
    clone_root = tmp_path / "repos"
    repo_dir = env_check.repo_dir_for_url(clone_root, "https://github.com/example/ready-model")
    (repo_dir / ".git").mkdir(parents=True)

    cache_dir = tmp_path / "cache" / "hfd"
    repo_cache = cache_dir / "models--org--model"
    snapshot = repo_cache / "snapshots" / "abc123"
    blob = repo_cache / "blobs" / "blob-a"
    (repo_cache / "refs").mkdir(parents=True)
    (repo_cache / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_cache / "refs" / "main").write_text("abc123", encoding="utf-8")
    blob.write_text("weights", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/blob-a")
    monkeypatch.setattr(env_check.sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(env_check, "check_command", lambda name: {"ok": True, "name": name, "path": f"/usr/bin/{name}"})

    result = env_check.check_manifest(
        manifest,
        clone_root=clone_root,
        cache_dir=cache_dir,
        require_repo=True,
        require_checkpoint=True,
        require_demo=True,
        require_runner_demo=False,
    )

    assert result["ok"] is True
    assert result["repo_checks"][0]["ready"] is True
    assert result["checkpoint_checks"][0]["ready"] is True
    assert result["demo_command_present"] is True
    assert result["runner_demo_command_present"] is False
    assert result["expected_artifacts"] == ["demo.mp4"]


def test_model_env_check_accepts_project_supported_python_range(monkeypatch: pytest.MonkeyPatch) -> None:
    env_check = _load_script("env_check")

    monkeypatch.setattr(env_check.sys, "version_info", (3, 11, 14))
    monkeypatch.setattr(env_check, "check_command", lambda name: {"ok": True, "name": name, "path": f"/usr/bin/{name}"})

    tools = env_check._tool_checks(require_hf=False)

    assert tools["python_supported"]["ok"] is True
    assert tools["python_supported"]["required"] == ">=3.10,<3.14"


def test_open_source_infer_repro_script_accepts_empty_clean_clone_cache(tmp_path: Path) -> None:
    open_source_infer_repro = _load_script("open_source_infer_repro")

    report = open_source_infer_repro.build_report(
        model_id="matrix-game-2",
        manifest_dir=REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog",
        cache_dir=tmp_path / "empty_hfd",
        output_dir=tmp_path / "report",
        strict_local=False,
    )

    assert report["ok"] is True
    assert report["local_ready"] is False
    assert report["docs"]["ok"] is True
    assert report["download_plan"]["ok"] is True
    assert report["symlink_layouts"]["ok"] is True
    assert report["local_check"]["ok"] is False


def test_model_zoo_scripts_are_stdlib_only() -> None:
    allowed_modules = set(sys.stdlib_module_names) | {"__future__", "worldfoundry", "yaml"}
    for path in (REPO_ROOT / "scripts" / "model_zoo").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                modules = {node.module.split(".", 1)[0]} if node.module else set()
            else:
                continue

            unexpected = modules - allowed_modules
            assert unexpected == set(), f"{path} imports {unexpected}"
