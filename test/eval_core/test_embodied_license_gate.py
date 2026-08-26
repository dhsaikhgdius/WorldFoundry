"""EM-02: embodied ``--accept-license`` flag + docker license env passthrough.

The license gate (``ensure_license``) has always hinted at
``--accept-license`` for non-interactive runs, but the CLI never grew the
flag, and docker children never inherited ``WORLDFOUNDRY_ACCEPTED_LICENSES``
so the in-container gate re-prompted (and exited) even after acceptance.
"""

from __future__ import annotations

import io
import os

import pytest

from worldfoundry.cli import _build_parser
from worldfoundry.evaluation.tasks.embodied.docker_runner import build_docker_run_command
from worldfoundry.evaluation.tasks.embodied.simulators.dirs import (
    ACCEPTED_LICENSES_ENV,
    accept_licenses,
    ensure_license,
)

# ── ensure_license gate ──────────────────────────────────────────


def test_ensure_license_passes_when_env_contains_id(monkeypatch) -> None:
    monkeypatch.setenv(ACCEPTED_LICENSES_ENV, "other, behavior1k ")
    ensure_license("behavior1k", url="https://example.com/licence", description="test licence")


def test_ensure_license_noninteractive_exits_with_flag_hint(monkeypatch, capsys) -> None:
    monkeypatch.delenv(ACCEPTED_LICENSES_ENV, raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as excinfo:
        ensure_license("behavior1k", url="https://example.com/licence", description="test licence")
    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "--accept-license behavior1k" in stderr
    assert ACCEPTED_LICENSES_ENV in stderr


# ── accept_licenses helper ───────────────────────────────────────


def test_accept_licenses_merges_and_dedupes(monkeypatch) -> None:
    monkeypatch.setenv(ACCEPTED_LICENSES_ENV, "existing")
    merged = accept_licenses(["behavior1k", "existing", " behavior1k ", ""])
    assert merged == "existing,behavior1k"
    assert os.environ[ACCEPTED_LICENSES_ENV] == "existing,behavior1k"


def test_accept_licenses_from_empty_env(monkeypatch) -> None:
    monkeypatch.delenv(ACCEPTED_LICENSES_ENV, raising=False)
    assert accept_licenses(["a", "b"]) == "a,b"
    assert os.environ[ACCEPTED_LICENSES_ENV] == "a,b"


def test_accept_licenses_noop_leaves_env_unset(monkeypatch) -> None:
    monkeypatch.delenv(ACCEPTED_LICENSES_ENV, raising=False)
    assert accept_licenses([]) == ""
    assert ACCEPTED_LICENSES_ENV not in os.environ


def test_accept_licenses_supports_explicit_mapping() -> None:
    env: dict[str, str] = {}
    assert accept_licenses(["x"], env=env) == "x"
    assert env == {ACCEPTED_LICENSES_ENV: "x"}


# ── CLI flag ─────────────────────────────────────────────────────


def test_embodied_run_parser_accepts_repeatable_flag(tmp_path) -> None:
    args = _build_parser().parse_args(
        [
            "embodied",
            "run",
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--accept-license",
            "behavior1k",
            "--accept-license",
            "robotwin",
            "--no-docker",
        ]
    )
    assert args.accept_license == ["behavior1k", "robotwin"]


def test_embodied_run_flag_defaults_to_none(tmp_path) -> None:
    args = _build_parser().parse_args(["embodied", "run", "--config", str(tmp_path / "cfg.yaml")])
    assert args.accept_license is None


def test_handle_embodied_run_merges_flag_into_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(ACCEPTED_LICENSES_ENV, raising=False)
    seen_env: dict[str, str | None] = {}

    class _StubEvaluateResult:
        status = "succeeded"
        sample_count = 0
        scorecard_path = tmp_path / "scorecard.json"
        exit_code = 0

    class _StubResult:
        evaluate_result = _StubEvaluateResult()

    def _stub_run(config_path, **kwargs):
        seen_env["value"] = os.environ.get(ACCEPTED_LICENSES_ENV)
        return _StubResult()

    import worldfoundry.evaluation.tasks.embodied.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "run_embodied_eval_config", _stub_run)
    args = _build_parser().parse_args(
        [
            "embodied",
            "run",
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--accept-license",
            "behavior1k",
            "--no-docker",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    assert seen_env["value"] == "behavior1k"


# ── docker passthrough ───────────────────────────────────────────


def _docker_cmd(tmp_path) -> list[str]:
    docker_config_path = tmp_path / "eval_config.yaml"
    docker_config_path.write_text("id: test\n", encoding="utf-8")
    return build_docker_run_command(
        {"docker": {"image": "example/bench:v1"}},
        docker_config_path=docker_config_path,
        output_dir=tmp_path / "out",
    )


def test_docker_run_forwards_accepted_licenses_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ACCEPTED_LICENSES_ENV, "behavior1k")
    cmd = _docker_cmd(tmp_path)
    pairs = list(zip(cmd, cmd[1:]))
    assert ("-e", ACCEPTED_LICENSES_ENV) in pairs
    # Value stays out of argv: the container inherits it from the host env.
    assert not any(item.startswith(f"{ACCEPTED_LICENSES_ENV}=") for item in cmd)


def test_docker_run_omits_license_env_when_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(ACCEPTED_LICENSES_ENV, raising=False)
    cmd = _docker_cmd(tmp_path)
    assert ACCEPTED_LICENSES_ENV not in cmd
