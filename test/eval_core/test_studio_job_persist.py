"""Tests for StudioJobStore on-disk persistence and orphan reconciliation."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.studio.jobs import StudioJobStore


def test_studio_job_store_persists_and_reconciles_orphans(tmp_path: Path) -> None:
    state_dir = tmp_path / "studio_jobs"
    store = StudioJobStore(max_workers=1, state_dir=state_dir, max_terminal_jobs=10)

    def _run(job):
        return {"status": "ok"}

    job = store.submit_run(
        title="demo",
        model_id="m",
        display_name="M",
        action="infer",
        metadata={},
        run_callable=_run,
    )
    # Wait for completion.
    assert job._future is not None
    job._future.result(timeout=5)
    assert job.terminal
    assert (state_dir / f"{job.job_id}.json").is_file()
    assert (state_dir / "jobs.json").is_file()

    # Inject a fake non-terminal row and restore.
    index = state_dir / "jobs.json"
    payload = index.read_text(encoding="utf-8")
    import json

    data = json.loads(payload)
    data["jobs"].append(
        {
            "job_id": "studio-00099",
            "title": "orphan",
            "model_id": "m",
            "display_name": "M",
            "action": "infer",
            "job_type": "inference",
            "status": "running",
            "created_at": "2020-01-01T00:00:00+00:00",
            "started_at": "2020-01-01T00:00:01+00:00",
            "completed_at": None,
            "error": None,
            "metadata": {},
            "logs": [],
        }
    )
    index.write_text(json.dumps(data), encoding="utf-8")

    restored = StudioJobStore(max_workers=1, state_dir=state_dir, max_terminal_jobs=10)
    orphan = restored.get("studio-00099")
    assert orphan is not None
    assert orphan.status == "failed"
    assert "restart" in (orphan.error or "").lower()


def test_studio_job_store_prunes_terminal_jobs(tmp_path: Path) -> None:
    store = StudioJobStore(max_workers=1, state_dir=tmp_path / "sj", max_terminal_jobs=2)

    def _run(job):
        return {"status": "ok"}

    futures = []
    for idx in range(4):
        job = store.submit_run(
            title=f"j{idx}",
            model_id="m",
            display_name="M",
            action="infer",
            metadata={},
            run_callable=_run,
        )
        assert job._future is not None
        futures.append(job._future)
    for fut in futures:
        fut.result(timeout=5)
    assert len([job for job in store.list() if job.terminal]) <= 2
