"""LG-01 leftover: Studio job run_id + log_context binding."""

from __future__ import annotations

from worldfoundry.core.logging_setup import get_log_context
from worldfoundry.studio.jobs import StudioJobStore


def test_studio_job_submit_binds_run_id_and_log_context() -> None:
    store = StudioJobStore(max_workers=1)
    seen: dict[str, object] = {}

    def _run(job):  # noqa: ANN001
        seen["job_id"] = job.job_id
        seen["run_id"] = job.run_id
        seen["context"] = get_log_context()
        return {"status": "succeeded"}

    job = store.submit_run(
        title="demo",
        model_id="demo-model",
        display_name="Demo",
        action="infer",
        metadata={},
        run_callable=_run,
    )
    assert job._future is not None
    job._future.result(timeout=5)
    assert job.run_id == job.job_id
    assert seen["run_id"] == job.job_id
    assert seen["context"]["run_id"] == job.job_id
    assert seen["context"]["job_id"] == job.job_id
    assert seen["context"]["phase"] == "studio_job"
    assert job.status == "completed"
