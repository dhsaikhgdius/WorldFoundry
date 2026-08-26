"""In-process job queue and lifecycle tracking for the WorldFoundry Studio UI.

Studio submits heavyweight inference/evaluation work as background jobs. This
module defines the job record (``StudioJob``), a thread-pool-backed store
(``StudioJobStore``), and Gradio-facing formatters for job tables and detail
panels.

The workspace app configures the worker count so users can run independent
inference jobs in parallel when enough GPUs are available.
"""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any

from worldfoundry.core.time import utc_now_iso
from worldfoundry.runtime.jobs import TERMINAL_JOB_STATUSES


STUDIO_JOB_TABLE_HEADERS = ["Job ID", "Title", "Model", "Action", "Status", "Created", "Elapsed"]
DEFAULT_STUDIO_MAX_TRACKED_JOBS = 500
_STUDIO_JOB_STATE_SCHEMA = "worldfoundry-studio-job-store-v1"
_PERSISTED_STUDIO_JOB_FIELDS = (
    "job_id",
    "title",
    "model_id",
    "display_name",
    "action",
    "job_type",
    "status",
    "created_at",
    "started_at",
    "completed_at",
    "error",
)


@dataclass
class StudioJob:
    """One Studio background job with status, logs, and optional result payload."""

    job_id: str
    title: str
    model_id: str
    display_name: str
    action: str
    job_type: str = "inference"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: Any | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    _future: Future[Any] | None = field(default=None, repr=False)
    _started_monotonic: float | None = field(default=None, repr=False)
    _completed_monotonic: float | None = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        """Return True when the job has reached a final status (completed/failed/cancelled)."""
        return self.status in TERMINAL_JOB_STATUSES

    def append_log(self, stream: str, text: str) -> None:
        """Append a timestamped log line from stdout/stderr/system."""
        if text:
            self.logs.append({"time": utc_now_iso(), "stream": stream, "text": text})

    def log_text(self, *, limit: int | None = None) -> str:
        """Render recent log lines as plain text for the UI log panel."""
        rows = self.logs[-limit:] if limit is not None else self.logs
        return "".join(f"[{row.get('stream', 'log')}] {row.get('text', '')}" for row in rows)

    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds since the job started (or 0 if not started)."""
        if self._started_monotonic is None:
            return 0.0
        end = self._completed_monotonic if self._completed_monotonic is not None else monotonic()
        return max(0.0, end - self._started_monotonic)


RunJobCallable = Callable[[StudioJob], Any]


def _result_indicates_failure(result: Any) -> tuple[bool, str | None]:
    """Inspect a run result and decide whether it should mark the job as failed.

    Runtime requirement plans are useful artifacts, but a ``blocked`` result
    must not be shown as a completed inference job in Studio.
    """
    artifact_kind = ""
    plan_path = ""
    if isinstance(result, Mapping):
        status = str(result.get("status") or "").strip().lower()
        detail = result.get("error") or result.get("blocked_reason")
        artifact_kind = str(result.get("artifact_kind") or "").strip().lower()
        plan_path = str(result.get("plan_path") or result.get("artifact_path") or "").strip()
    else:
        status = str(getattr(result, "status", "") or "").strip().lower()
        detail = None
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping):
            model_result = metadata.get("result")
            if isinstance(model_result, Mapping):
                detail = model_result.get("error") or model_result.get("blocked_reason")
                artifact_kind = str(model_result.get("artifact_kind") or "").strip().lower()
                plan_path = str(model_result.get("plan_path") or model_result.get("artifact_path") or "").strip()
    # Treat any non-success terminal status as a job failure for the UI.
    if status and status not in {
        "ok",
        "succeeded",
        "completed",
        "planned",
        "normalized",
        "official_verified",
        "official_results_imported",
        "official_results_normalized",
        "official_result_normalization",
        "official_results_normalization",
    }:
        suffix = f": {detail}" if detail else ""
        return True, f"run status: {status}{suffix}"
    return False, None


class StudioJobStore:
    """In-process job runner for the unified Studio UI.

    Defaults to a small worker pool. Terminal jobs are pruned when
    ``max_jobs`` is set so long-lived Workspace processes cannot grow without
    bound. Optional ``state_path`` persists job metadata for UI recovery after
    restart (thread-pool work itself is not resumed).
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_log_lines: int = 2000,
        initial_counter: int = 0,
        max_jobs: int | None = DEFAULT_STUDIO_MAX_TRACKED_JOBS,
        state_path: str | Path | None = None,
    ) -> None:
        """Create a job store backed by a small thread pool (default: one worker)."""
        self.max_log_lines = max_log_lines
        self.max_jobs = max_jobs
        self.state_path = None if state_path is None else Path(state_path)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="worldfoundry-studio-job")
        self._jobs: dict[str, StudioJob] = {}
        self._lock = RLock()
        self._counter = max(0, initial_counter)
        if self.state_path is not None:
            self._restore_persisted_jobs()

    def _restore_persisted_jobs(self) -> None:
        assert self.state_path is not None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return
        rows = payload.get("jobs") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("job_id"):
                continue
            job = StudioJob(
                job_id=str(row["job_id"]),
                title=str(row.get("title") or row["job_id"]),
                model_id=str(row.get("model_id") or ""),
                display_name=str(row.get("display_name") or row.get("model_id") or ""),
                action=str(row.get("action") or ""),
                job_type=str(row.get("job_type") or "inference"),
            )
            for field_name in _PERSISTED_STUDIO_JOB_FIELDS:
                if field_name == "job_id" or row.get(field_name) is None:
                    continue
                setattr(job, field_name, row[field_name])
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping):
                job.metadata = dict(metadata)
            # Thread-pool work cannot be resumed; mark non-terminal restored rows failed.
            if not job.terminal:
                job.status = "failed"
                job.error = job.error or "job lost during Studio restart (in-process worker not restored)"
                job.completed_at = job.completed_at or utc_now_iso()
            self._jobs[job.job_id] = job
            suffix = job.job_id.removeprefix("studio-")
            if suffix.isdigit():
                self._counter = max(self._counter, int(suffix))

    def _persist(self) -> None:
        if self.state_path is None:
            return
        rows = []
        for job in sorted(self._jobs.values(), key=lambda item: item.created_at):
            row: dict[str, Any] = {name: getattr(job, name) for name in _PERSISTED_STUDIO_JOB_FIELDS}
            row["metadata"] = dict(job.metadata)
            rows.append(row)
        payload = {"schema_version": _STUDIO_JOB_STATE_SCHEMA, "jobs": rows}
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_name(self.state_path.name + f".tmp-{os.getpid()}")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(tmp_path, self.state_path)
        except OSError:
            pass

    def prune(self, *, max_jobs: int | None = None) -> int:
        """Evict oldest terminal jobs when the store exceeds ``max_jobs``."""
        with self._lock:
            return self._prune_unlocked(max_jobs=max_jobs)

    def _prune_unlocked(self, *, max_jobs: int | None = None) -> int:
        limit = self.max_jobs if max_jobs is None else max_jobs
        if limit is None:
            return 0
        if len(self._jobs) <= limit:
            return 0
        removed = 0
        terminal = sorted(
            (job for job in self._jobs.values() if job.terminal),
            key=lambda job: job.created_at,
        )
        excess = len(self._jobs) - limit
        for job in terminal:
            if excess <= 0:
                break
            self._jobs.pop(job.job_id, None)
            removed += 1
            excess -= 1
        if removed:
            self._persist()
        return removed

    def submit_run(
        self,
        *,
        title: str,
        model_id: str,
        display_name: str,
        action: str,
        job_type: str = "inference",
        metadata: Mapping[str, Any] | None,
        run_callable: RunJobCallable,
    ) -> StudioJob:
        """Enqueue a new job and execute ``run_callable(job)`` on the thread pool."""
        with self._lock:
            if self.max_jobs is not None:
                self._prune_unlocked(max_jobs=self.max_jobs)
            self._counter += 1
            job_id = f"studio-{self._counter:05d}"
            job = StudioJob(
                job_id=job_id,
                title=title or f"{display_name} {action}",
                model_id=model_id,
                display_name=display_name,
                action=action,
                job_type=job_type,
                metadata=dict(metadata or {}),
            )
            self._jobs[job.job_id] = job
            job._future = self._executor.submit(self._execute, job.job_id, run_callable)
            self._persist()
            return job

    def get(self, job_id: str | None) -> StudioJob | None:
        """Return a job by id, or None when id is empty or unknown."""
        if not job_id:
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[StudioJob]:
        """Return all jobs sorted by creation time (newest first)."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def cancel(self, job_id: str | None) -> tuple[bool, str]:
        """Request cancellation; queued jobs cancel immediately, running jobs stop cooperatively."""
        job = self.get(job_id)
        if job is None:
            return False, "select a job first"
        with self._lock:
            if job.terminal:
                return False, f"job is already {job.status}"
            job.cancel_requested = True
            future = job._future
            if future is not None and future.cancel():
                job.status = "cancelled"
                job.error = "cancelled before start"
                job.completed_at = utc_now_iso()
                job._completed_monotonic = monotonic()
                job.append_log("system", "cancelled before start\n")
                self._persist()
                return True, "cancelled"
            job.append_log("system", "cancellation requested; active inference will stop after the current call returns\n")
            self._persist()
            return True, "cancellation requested"

    def _execute(self, job_id: str, run_callable: RunJobCallable) -> None:
        """Worker entrypoint: transition job state, invoke callable, capture errors/logs."""
        job = self.get(job_id)
        if job is None:
            return
        with self._lock:
            if job.cancel_requested:
                job.status = "cancelled"
                job.error = "cancelled before start"
                job.completed_at = utc_now_iso()
                job._completed_monotonic = monotonic()
                return
            job.status = "running"
            job.started_at = utc_now_iso()
            job._started_monotonic = monotonic()
            job.append_log("system", f"started {job.title}\n")

        try:
            result = run_callable(job)
            with self._lock:
                job.result = result
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.error = "cancelled after current inference completed"
                else:
                    failed_result, error = _result_indicates_failure(result)
                    if failed_result:
                        job.status = "failed"
                        job.error = error
                        job.append_log("system", f"{error}\n")
                    else:
                        job.status = "completed"
                        job.append_log("system", "completed\n")
        except Exception as exc:  # noqa: BLE001 - surfaced through local UI.
            with self._lock:
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.error = f"cancelled: {type(exc).__name__}: {exc}"
                    job.append_log("system", "cancelled active job\n")
                else:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.append_log("stderr", traceback.format_exc())
        finally:
            with self._lock:
                if len(job.logs) > self.max_log_lines:
                    del job.logs[: len(job.logs) - self.max_log_lines]
                job.completed_at = job.completed_at or utc_now_iso()
                job._completed_monotonic = monotonic()
                self._persist()


def format_elapsed(job: StudioJob) -> str:
    """Format job elapsed time as a compact human-readable string."""
    seconds = int(job.elapsed_seconds())
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def format_job_choices(jobs: Sequence[StudioJob]) -> list[tuple[str, str]]:
    """Build Gradio dropdown choices mapping label -> job_id."""
    return [
        (f"{job.title} · {job.status} · {job.job_id}", job.job_id)
        for job in jobs
    ]


def format_jobs_table(jobs: Sequence[StudioJob]) -> list[list[str]]:
    """Build row data for the Studio jobs table widget."""
    rows: list[list[str]] = []
    for job in jobs:
        rows.append(
            [
                job.job_id,
                job.title,
                job.display_name,
                job.action,
                job.status,
                job.created_at,
                format_elapsed(job),
            ]
        )
    return rows


def job_detail_html(job: StudioJob | None) -> str:
    """Render HTML summary panel for one job (model, prompt, output, error)."""
    if job is None:
        return '<div class="wa-job-detail wa-job-empty">No Studio jobs yet.</div>'
    metadata = dict(job.metadata)
    prompt = str(metadata.get("prompt") or "").strip()
    prompt_html = f"<p><strong>Prompt</strong><br>{escape(prompt)}</p>" if prompt else ""
    output_dir = ""
    result = job.result
    if result is not None:
        output_dir = str(getattr(result, "output_dir", "") or "")
    if not output_dir:
        output_dir = str(metadata.get("output_root") or "")
    output_html = f"<p><strong>Output</strong><br><code>{escape(output_dir)}</code></p>" if output_dir else ""
    error_html = f"<p class=\"wa-job-error\"><strong>Error</strong><br>{escape(job.error or '')}</p>" if job.error else ""
    return f"""
<div class="wa-job-detail wa-job-status-{escape(job.status)}">
  <div class="wa-job-detail-head">
    <strong>{escape(job.title)}</strong>
    <span>{escape(job.status)}</span>
  </div>
  <p><strong>Model</strong><br>{escape(job.display_name)} <code>{escape(job.model_id)}</code></p>
  <p><strong>Action</strong><br>{escape(job.action)} · {escape(job.job_type)}</p>
  {prompt_html}
  {output_html}
  {error_html}
</div>
"""


def job_log_text(job: StudioJob | None, *, limit: int = 120) -> str:
    """Return truncated plain-text logs for the selected job."""
    if job is None:
        return ""
    return job.log_text(limit=limit)


__all__ = [
    "STUDIO_JOB_TABLE_HEADERS",
    "StudioJob",
    "StudioJobStore",
    "format_job_choices",
    "format_jobs_table",
    "job_detail_html",
    "job_log_text",
]
