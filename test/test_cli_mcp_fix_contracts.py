"""Regression tests for the CLI/MCP review fixes (plan/code_review/05_cli_mcp.md).

Covers the contracts introduced by the CM-01/07/25/26/29/31/32 repairs:
startup-path laziness, structured ``--json`` errors, the MCP error envelope,
async studio waits with finite defaults, absolute MCP default paths, tool-face
consistency, and the persistent ``MCPClient`` session.

CPU-only; no GPU, network, or optional third-party packages required.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "worldfoundry.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=timeout,
    )


class FakeMCP:
    """Minimal stand-in for FastMCP that records registered tool functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture(scope="module")
def registered_tools() -> dict[str, object]:
    from worldfoundry.mcp.tools.registration import register_tools

    fake = FakeMCP()
    register_tools(fake)
    return fake.tools


# ── CM-01: query paths must not import torch or the orchestration stack ──


def test_parser_build_stays_off_heavy_import_paths() -> None:
    # ``run_mode``/``runtime_preflight`` are light parser-data leaves and stay
    # eager; torch and the ~0.7s ``orchestration.service`` chain must not load.
    code = (
        "import sys\n"
        "from worldfoundry.cli.main import _build_parser\n"
        "_build_parser()\n"
        "assert 'torch' not in sys.modules, 'torch imported during parser build'\n"
        "heavy = 'worldfoundry.evaluation.tasks.execution.orchestration.service'\n"
        "assert heavy not in sys.modules, 'orchestration.service imported during parser build'\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_help_exits_zero_without_torch() -> None:
    result = _run_cli(["--help"])
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


# ── CM-07: --json failures keep stdout parseable and exit 1 ─────────────


def test_json_error_contract_and_exit_code() -> None:
    result = _run_cli(["zoo", "benchmark-show", "--benchmark-id", "does-not-exist-xyz", "--json"])
    assert result.returncode == 1, (result.returncode, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["error"]["type"]
    assert "does-not-exist-xyz" in payload["error"]["message"]
    # Human-facing summary goes to stderr, without a traceback by default.
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_usage_error_keeps_exit_code_two() -> None:
    result = _run_cli(["definitely-not-a-command"])
    assert result.returncode == 2
    assert "usage:" in result.stderr


# ── CM-25: registered tools return the structured error envelope ────────


def test_tool_errors_return_envelope_instead_of_raising(registered_tools: dict[str, object]) -> None:
    list_runs = registered_tools["list_runs"]
    payload = list_runs(limit=0)
    assert payload["ok"] is False
    assert payload["error_type"] == "error"
    assert "limit" in payload["error"]


def test_async_tool_errors_return_envelope(registered_tools: dict[str, object]) -> None:
    evaluate = registered_tools["evaluate"]
    payload = asyncio.run(evaluate(model="m", benchmark="b", wait="bogus-strategy"))
    assert payload["ok"] is False
    assert payload["error_type"] == "error"
    assert "wait" in payload["error"]


# ── CM-26: studio waits are async with finite default timeouts ──────────


def test_studio_wait_tools_are_async(registered_tools: dict[str, object]) -> None:
    assert inspect.iscoroutinefunction(registered_tools["wait_for_studio_job"])
    assert inspect.iscoroutinefunction(registered_tools["submit_studio_inference"])


def test_studio_wait_default_timeout_is_finite() -> None:
    from worldfoundry.mcp.tools import studio

    assert studio.DEFAULT_STUDIO_WAIT_TIMEOUT_S == 600.0
    for fn, param in (
        (studio.wait_for_studio_job_payload, "timeout_s"),
        (studio.wait_for_studio_job_payload_async, "timeout_s"),
        (studio.submit_studio_inference_payload, "wait_timeout_s"),
        (studio.submit_studio_inference_payload_async, "wait_timeout_s"),
    ):
        default = inspect.signature(fn).parameters[param].default
        assert default == studio.DEFAULT_STUDIO_WAIT_TIMEOUT_S


# ── CM-29: MCP defaults resolve to absolute paths ────────────────────────


def test_mcp_default_output_root_is_absolute() -> None:
    from worldfoundry.mcp.tools.context import DEFAULT_MCP_OUTPUT_ROOT

    assert DEFAULT_MCP_OUTPUT_ROOT.is_absolute()


def test_readiness_default_dataset_root_is_absolute() -> None:
    from worldfoundry.mcp.tools.readiness import _default_dataset_root

    assert _default_dataset_root().is_absolute()


# ── CM-31: advertised tool face matches the registrations ────────────────


def test_mcp_tool_names_match_registration(registered_tools: dict[str, object]) -> None:
    from worldfoundry.mcp.tools.server_info import MCP_TOOL_NAMES

    assert sorted(MCP_TOOL_NAMES) == sorted(registered_tools)


def test_preview_run_covers_evaluate_selection_surface(registered_tools: dict[str, object]) -> None:
    preview_params = set(inspect.signature(registered_tools["preview_run"]).parameters)
    evaluate_params = set(inspect.signature(registered_tools["evaluate"]).parameters)
    # Wait-strategy knobs only make sense for submission, not preview.
    assert evaluate_params - preview_params <= {"wait", "wait_timeout_s"}


# ── CM-32: MCPClient supports a persistent session ────────────────────────


def test_mcp_client_persistent_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.mcp.client import MCPClient

    created: list[object] = []

    class FakeSession:
        async def list_tools(self):
            class Result:
                tools = ["one"]

            return Result()

        async def call_tool(self, name, arguments):
            return {"name": name, "arguments": arguments}

    class FakeContext:
        def __init__(self) -> None:
            self.closed = False

        async def __aenter__(self):
            created.append(self)
            return FakeSession()

        async def __aexit__(self, *exc):
            self.closed = True
            return False

    async def fake_client(self):
        return FakeContext()

    monkeypatch.setattr(MCPClient, "_client", fake_client)

    async def scenario() -> None:
        # Default mode still opens one session per call.
        client = MCPClient()
        await client.list_tools()
        await client.call_tool("x")
        assert len(created) == 2
        assert all(ctx.closed for ctx in created)

        created.clear()
        async with MCPClient() as persistent:
            await persistent.list_tools()
            await persistent.call_tool("y")
            await persistent.call_tool("z")
        assert len(created) == 1
        assert created[0].closed

    asyncio.run(scenario())
