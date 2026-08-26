"""LG-02: contract generation emits sample_id-scoped structured events."""

from __future__ import annotations

import logging

from worldfoundry.core.logging_setup import configure_logging
from worldfoundry.evaluation.api import GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.execution.orchestration.contract import _generate


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _StubRunner:
    model_id = "stub-model"

    def generate(self, requests):  # noqa: ANN001 - test double
        return [
            GenerationResult(
                sample_id=request.sample_id,
                request_id=request.request_id,
                model_id=self.model_id,
                status="succeeded",
            )
            for request in requests
        ]


def test_generate_emits_sample_generation_finished_events() -> None:
    configure_logging(level="INFO", force=True)
    handler = _RecordingHandler()
    target = logging.getLogger("worldfoundry.evaluation.tasks.execution.orchestration.contract")
    target.addHandler(handler)
    try:
        requests = [
            GenerationRequest(sample_id="sample-0001", task_name="demo", split="test", request_id="req-1"),
            GenerationRequest(sample_id="sample-0002", task_name="demo", split="test", request_id="req-2"),
        ]
        results = _generate(_StubRunner(), requests)
        assert [result.sample_id for result in results] == ["sample-0001", "sample-0002"]

        events = [
            getattr(record, "_worldfoundry_fields", {})
            for record in handler.records
            if getattr(record, "_worldfoundry_fields", {}).get("event") == "sample.generation.finished"
        ]
        assert [event.get("sample_id") for event in events] == ["sample-0001", "sample-0002"]
        assert all(event.get("status") == "succeeded" for event in events)
    finally:
        target.removeHandler(handler)
