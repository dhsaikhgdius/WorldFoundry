from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ACQUIRE_SCRIPT = REPO_ROOT / "scripts" / "vla_va_wam_acquire.py"

# DS-03: scripts/vla_va_wam_acquire.py is not shipped in-tree yet; restore before
# re-enabling this module as a hard release gate.
pytestmark = pytest.mark.skipif(
    not ACQUIRE_SCRIPT.is_file(),
    reason="scripts/vla_va_wam_acquire.py missing (DS-03); restore script before re-enabling this gate",
)


def _load_script() -> ModuleType:
    path = ACQUIRE_SCRIPT
    spec = importlib.util.spec_from_file_location("test_vla_va_wam_acquire_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, targets: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"targets": targets}), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_repo_slug_includes_owner_to_avoid_cache_collisions() -> None:
    acquire = _load_script()

    assert acquire.repo_slug("https://github.com/org/project.git") == "org--project"
    assert acquire.repo_slug("git@github.com:other/project.git") == "other--project"


def test_plan_only_clone_writes_item_log_report_and_summary(tmp_path: Path) -> None:
    acquire = _load_script()
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [
            {
                "id": "alpha",
                "groups": ["p0"],
                "priority": "P0",
                "official_sources": {"github": [{"url": "https://github.com/example/project", "clone": True}]},
            }
        ],
    )
    report = tmp_path / "report.jsonl"
    summary = tmp_path / "summary.json"

    rc = acquire.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path / "cache"),
            "--target",
            "alpha",
            "--clone",
            "--plan-only",
            "--report-jsonl",
            str(report),
            "--summary-json",
            str(summary),
            "--timeout-seconds",
            "5",
            "--retries",
            "1",
        ]
    )

    assert rc == 0
    events = _read_jsonl(report)
    assert len(events) == 1
    assert events[0]["status"] == "success"
    assert events[0]["kind"] == "github"
    assert events[0]["attempts"] == 1
    assert events[0]["command"][:2] == ["git", "clone"]
    assert events[0]["path_exists"] is False
    assert events[0]["path_nonempty"] is False
    assert Path(str(events[0]["log_path"])).read_text(encoding="utf-8").startswith("[plan-only] git clone")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["counts"] == {"success": 1}


def test_hf_token_is_redacted_from_report_and_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acquire = _load_script()
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [
            {
                "id": "model-target",
                "groups": ["vla"],
                "priority": "P0",
                "official_sources": {"huggingface_models": [{"repo_id": "org/model"}]},
            }
        ],
    )
    hfd_script = tmp_path / "hfd.sh"
    hfd_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(acquire.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    report = tmp_path / "report.jsonl"

    rc = acquire.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path / "cache"),
            "--target",
            "model-target",
            "--download-models",
            "--plan-only",
            "--hfd-script",
            str(hfd_script),
            "--report-jsonl",
            str(report),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert rc == 0
    report_text = report.read_text(encoding="utf-8")
    assert "hf_secret_token" not in report_text
    assert "--hf_token" not in report_text
    event = _read_jsonl(report)[0]
    log_text = Path(str(event["log_path"])).read_text(encoding="utf-8")
    assert "hf_secret_token" not in log_text
    assert "--hf_token" not in log_text


def test_optional_huge_requires_include_huge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acquire = _load_script()
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [
            {
                "id": "huge-data",
                "groups": ["p1"],
                "priority": "P1",
                "official_sources": {
                    "huggingface_datasets": [{"repo_id": "org/huge", "download": "optional_huge"}]
                },
            }
        ],
    )
    hfd_script = tmp_path / "hfd.sh"
    hfd_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(acquire.shutil, "which", lambda name: f"/usr/bin/{name}")
    report = tmp_path / "report.jsonl"

    rc = acquire.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path / "cache"),
            "--target",
            "huge-data",
            "--download-datasets",
            "--include-large",
            "--plan-only",
            "--hfd-script",
            str(hfd_script),
            "--report-jsonl",
            str(report),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert rc == 0
    event = _read_jsonl(report)[0]
    assert event["status"] == "skipped"
    assert event["reason"] == "optional_huge"


def test_hf_download_skips_existing_shared_hfd_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acquire = _load_script()
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [
            {
                "id": "existing-model",
                "groups": ["p1"],
                "priority": "P1",
                "official_sources": {"huggingface_models": [{"repo_id": "org/model"}]},
            }
        ],
    )
    hfd_script = tmp_path / "hfd.sh"
    hfd_script.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    shared_hfd = tmp_path / "shared_hfd"
    cached_repo = shared_hfd / "org--model"
    cached_repo.mkdir(parents=True)
    (cached_repo / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(acquire.shutil, "which", lambda name: f"/usr/bin/{name}")
    report = tmp_path / "report.jsonl"

    rc = acquire.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path / "cache"),
            "--target",
            "existing-model",
            "--download-models",
            "--hf-models-root",
            str(shared_hfd),
            "--hfd-script",
            str(hfd_script),
            "--report-jsonl",
            str(report),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert rc == 0
    event = _read_jsonl(report)[0]
    assert event["status"] == "skipped"
    assert event["reason"] == "already_exists"
    assert event["path"] == str(cached_repo)
    assert event["path_nonempty"] is True


def test_write_plan_records_custom_hf_roots(tmp_path: Path) -> None:
    acquire = _load_script()
    manifest = _write_manifest(
        tmp_path / "targets.json",
        [{"id": "alpha", "groups": ["p0"], "priority": "P0"}],
    )
    plan = tmp_path / "plan.json"
    shared_models = tmp_path / "hfd_models"
    shared_datasets = tmp_path / "hfd_datasets"

    rc = acquire.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path / "cache"),
            "--target",
            "alpha",
            "--hf-models-root",
            str(shared_models),
            "--hf-datasets-root",
            str(shared_datasets),
            "--write-plan",
            str(plan),
        ]
    )

    assert rc == 0
    payload = json.loads(plan.read_text(encoding="utf-8"))
    assert payload["options"]["hf_models_root"] == str(shared_models)
    assert payload["options"]["hf_datasets_root"] == str(shared_datasets)


def test_run_command_retries_until_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acquire = _load_script()
    calls: list[object] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(acquire.subprocess, "run", fake_run)

    result = acquire.run_command(
        ["tool", "arg"],
        env={},
        plan_only=False,
        timeout_seconds=7,
        retries=2,
        log_path=tmp_path / "item.log",
        secrets=[],
    )

    assert result.returncode == 0
    assert result.attempts == 2
    assert calls == [7, 7]
    assert "[retry] command failed; retrying" in (tmp_path / "item.log").read_text(encoding="utf-8")


def test_run_command_redacts_token_from_subprocess_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acquire = _load_script()
    secret = "hf_output_secret"

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=f"using {secret}\n")

    monkeypatch.setattr(acquire.subprocess, "run", fake_run)

    result = acquire.run_command(
        ["tool", "--hf_token", secret],
        env={},
        plan_only=False,
        timeout_seconds=7,
        retries=0,
        log_path=tmp_path / "output.log",
        secrets=[secret],
    )

    log_text = (tmp_path / "output.log").read_text(encoding="utf-8")
    assert result.returncode == 0
    assert secret not in log_text
    assert "--hf_token <redacted>" in log_text
    assert "using <redacted>" in log_text


def test_run_command_reports_timeout_after_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acquire = _load_script()

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr(acquire.subprocess, "run", fake_run)

    result = acquire.run_command(
        ["slow-tool"],
        env={},
        plan_only=False,
        timeout_seconds=3,
        retries=1,
        log_path=tmp_path / "timeout.log",
        secrets=[],
    )

    assert result.returncode == 124
    assert result.timed_out is True
    assert result.attempts == 2
    assert "timed out after 3 seconds" in str(result.error)


def test_post_check_requires_existing_nonempty_directory(tmp_path: Path) -> None:
    acquire = _load_script()

    missing = acquire.post_check_event("repos_root", tmp_path / "missing")
    assert missing["status"] == "failed"
    assert missing["exists"] is False
    assert missing["nonempty"] is False

    nonempty = tmp_path / "repos"
    nonempty.mkdir()
    (nonempty / "marker").write_text("ok", encoding="utf-8")
    present = acquire.post_check_event("repos_root", nonempty)
    assert present["status"] == "success"
    assert present["exists"] is True
    assert present["nonempty"] is True
