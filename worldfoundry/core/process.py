"""Import-light subprocess launch helpers shared by model runtimes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def terminate_process_group(process: subprocess.Popen[Any], *, grace_seconds: float = 3.0) -> None:
    """Terminate a process and all descendants in its dedicated process group."""

    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=max(float(grace_seconds), 0.0))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    process.wait()


def terminate_process_tree(process: subprocess.Popen[Any], *, grace_seconds: float = 3.0) -> None:
    """Terminate one process and recursively terminate descendants without a process group."""

    if process.poll() is not None:
        return
    descendants: list[Any] = []
    try:
        import psutil
    except ImportError:
        descendants = []
    else:
        try:
            parent = psutil.Process(process.pid)
            descendants = parent.children(recursive=True)
            for child in reversed(descendants):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
        except (OSError, psutil.AccessDenied, psutil.NoSuchProcess):
            descendants = []
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=max(float(grace_seconds), 0.0))
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
        process.wait()
    if descendants:
        import psutil

        _, alive = psutil.wait_procs(descendants, timeout=max(float(grace_seconds), 0.0))
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if alive:
            psutil.wait_procs(alive, timeout=max(float(grace_seconds), 0.0))


def synthesis_timeout_seconds(default: float | None = None) -> float | None:
    """Resolve ``WORLDFOUNDRY_SYNTHESIS_TIMEOUT_SECONDS`` with an optional default.

    Empty / unset keeps ``default``. Zero disables the deadline (returns ``None``).
    """

    raw = os.getenv("WORLDFOUNDRY_SYNTHESIS_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return default
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise ValueError(
            "WORLDFOUNDRY_SYNTHESIS_TIMEOUT_SECONDS must be a non-negative number of seconds."
        ) from exc
    if timeout_s < 0.0:
        raise ValueError(
            f"WORLDFOUNDRY_SYNTHESIS_TIMEOUT_SECONDS must be non-negative, got {timeout_s}."
        )
    return timeout_s or None


def run_logged_subprocess(
    command: str | Sequence[str],
    *,
    stdout_path: str | Path,
    stderr_path: str | Path,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
    append: bool = False,
    start_new_session: bool | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run a subprocess with bounded log memory and timeout-safe descendants.

    The child receives the current WorldFoundry correlation context and gets a
    dedicated JSONL lifecycle artifact beside its raw stdout/stderr captures.
    This keeps framework-owned workers observable without requiring changes to
    vendored official benchmark code.
    """

    from worldfoundry.core.logging_setup import log_context_environment, write_jsonl_event

    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    owns_process_group = os.name == "posix" and start_new_session is not False
    lifecycle_path = stdout_path.parent / "logs" / f"{stdout_path.stem}.events.jsonl"
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    process_env.update(log_context_environment())
    # Avoid concurrent workers rotating the parent CLI's event file.  A child
    # that enters WorldFoundry's CLI/configuration gets its own structured
    # worker sink; third-party scripts simply ignore these variables.
    process_env["WORLDFOUNDRY_LOG_FILE"] = str(lifecycle_path)
    process_env["WORLDFOUNDRY_LOG_JSON"] = "1"
    command_name = (
        command.split(maxsplit=1)[0]
        if isinstance(command, str) and command.strip()
        else str(command[0])
        if not isinstance(command, str) and command
        else ""
    )
    start = time.monotonic()
    write_jsonl_event(
        lifecycle_path,
        level="INFO",
        event="subprocess.started",
        message="WorldFoundry subprocess started",
        logger_name=__name__,
        executable=command_name,
        cwd=None if cwd is None else str(cwd),
        stdout_path=str(stdout_path.resolve()),
        stderr_path=str(stderr_path.resolve()),
        timeout_seconds=timeout,
    )
    with (
        stdout_path.open(mode, encoding="utf-8", errors="replace") as stdout,
        stderr_path.open(mode, encoding="utf-8", errors="replace") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=process_env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            shell=shell,
            start_new_session=owns_process_group,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if owns_process_group:
                terminate_process_group(process)
            else:
                terminate_process_tree(process)
            write_jsonl_event(
                lifecycle_path,
                level="ERROR",
                event="subprocess.timed_out",
                message="WorldFoundry subprocess timed out",
                logger_name=__name__,
                duration_seconds=round(time.monotonic() - start, 6),
                timeout_seconds=timeout,
            )
            raise subprocess.TimeoutExpired(command, timeout) from None
    write_jsonl_event(
        lifecycle_path,
        level="INFO" if returncode == 0 else "ERROR",
        event="subprocess.finished" if returncode == 0 else "subprocess.failed",
        message="WorldFoundry subprocess finished",
        logger_name=__name__,
        returncode=returncode,
        duration_seconds=round(time.monotonic() - start, 6),
    )
    return subprocess.CompletedProcess(command, returncode)


def read_text_tail(
    path: str | Path,
    *,
    max_lines: int = 20,
    max_bytes: int = 64 * 1024,
) -> str:
    """Read a bounded UTF-8 tail from a potentially very large log file."""

    if max_lines < 1 or max_bytes < 1:
        raise ValueError("max_lines and max_bytes must be positive")
    resolved = Path(path)
    if not resolved.is_file():
        return ""
    with resolved.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        offset = max(size - int(max_bytes), 0)
        handle.seek(offset)
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    if offset:
        _, separator, text = text.partition("\n")
        if not separator:
            text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-int(max_lines) :])


def torchrun_module_command(
    module: str,
    *,
    nproc_per_node: int,
    args: Sequence[str] = (),
    python_executable: str = sys.executable,
) -> list[str]:
    """Build a single-node torchrun command for a Python module."""

    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be positive")
    return [
        python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        "-m",
        module,
        *map(str, args),
    ]


def run_torchrun_module(
    module: str,
    *,
    nproc_per_node: int,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    python_executable: str = sys.executable,
) -> subprocess.CompletedProcess[str]:
    """Run a single-node Python module under torchrun and capture its logs."""

    return subprocess.run(
        torchrun_module_command(
            module,
            nproc_per_node=nproc_per_node,
            args=args,
            python_executable=python_executable,
        ),
        env=None if env is None else dict(env),
        text=True,
        capture_output=True,
        check=False,
    )


__all__ = [
    "read_text_tail",
    "run_logged_subprocess",
    "run_torchrun_module",
    "synthesis_timeout_seconds",
    "terminate_process_group",
    "terminate_process_tree",
    "torchrun_module_command",
]
