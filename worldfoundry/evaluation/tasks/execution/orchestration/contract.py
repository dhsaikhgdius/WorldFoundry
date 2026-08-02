"""
Module for defining and executing a contract-based evaluation run.

This module provides the core logic for running a benchmark or evaluation
suite against a World Model Runner. It handles request normalization,
generation with caching, metric computation, aggregation, and artifact
management, including generating manifests, execution plans, and scorecards.
It defines the `ContractRunRequest` and `ContractRunResult` data structures
to formalize the input and output of an evaluation run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from worldfoundry.core.io.serialization import JsonlWriter, iter_jsonl
from worldfoundry.evaluation.api import (
    AggregateResult,
    GenerationRequest,
    GenerationResult,
    Metric,
    MetricResult,
    WorldModelRunner,
    align_batch_metric_outputs,
    enrich_artifact_ref,
    is_generation_result_successful,
)
from worldfoundry.evaluation.reporting import write_run_manifest_artifacts, write_run_report_artifacts
from worldfoundry.evaluation.reporting.scorecard import write_scorecard
from worldfoundry.evaluation.utils import (
    build_run_fingerprint,
    build_version_context,
    jsonable,
    reset_jsonl,
    write_json,
    write_jsonl,
)

from .cache import cache_paths_from_stats, generation_cache_hit_metadata, run_generation_with_cache

JsonRow = dict[str, Any]


@dataclass(frozen=True)
class ContractRunRequest:
    """Formalized batch request for live online execution via a specified Model Runner.
    
    Acts as the main payload definition bridging the orchestrator loop to actual WorldModelRunner implementations.
    Provides necessary configurations including metrics suite bindings and fallback constraints.
    """
    output_dir: str | Path
    requests: Sequence[GenerationRequest | Mapping[str, Any]]
    runner: WorldModelRunner
    metrics: Sequence[Metric] = ()
    benchmark: Mapping[str, Any] | Any | None = None
    model: Mapping[str, Any] | Any | None = None
    dataset: Mapping[str, Any] | Any | None = None
    run_id: str | None = None
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True
    cleanup_runner: bool = True
    resume: bool = False
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "contract_runner"


@dataclass(frozen=True)
class ContractRunResult:
    """Final structural summary returned after local runner task completion.

    Records the footprint of generated assets (like scorecards or execution plans), alongside 
    exit codes and sample metrics summary.
    """
    status: str
    exit_code: int
    output_dir: Path
    manifest_path: Path
    execution_plan_path: Path
    scorecard_path: Path
    sample_count: int
    successful_sample_count: int
    failed_sample_count: int
    artifact_count: int


class ContractRunner:
    """Minimal in-process runtime executing the strict GenerationRequest/Result protocol.

    This delegator drives the internal feedback loop: feeding tasks to a `WorldModelRunner`, 
    capturing its `GenerationResult`s, enriching emitted media artifacts via local caching rules,
    evaluating them against the attached metric suite, and logging the output JSONL streams.
    """

    def run(
        self,
        request: ContractRunRequest | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ContractRunResult:
        """Entrypoint for executing the localized pipeline against a bounded model."""
        return run_contract(request, **kwargs)


def _utcnow_iso() -> str:
    """Returns the current UTC datetime as an ISO 8601 string, without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[JsonRow]:
    """Reads a JSONL file and returns its content as a list of dictionaries.

    Args:
        path: The path to the JSONL file.

    Returns:
        A list of dictionaries, where each dictionary represents a JSON line.
    """
    if not path.exists():
        return []
    rows: list[JsonRow] = []
    for row in iter_jsonl(path):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _coerce_mapping(value: Any) -> JsonRow:
    """Coerces a value into a JSON-compatible dictionary.

    If the value is already a mapping, it's converted to a dictionary.
    Otherwise, it's wrapped in a dictionary with a 'value' key.

    Args:
        value: The value to coerce.

    Returns:
        A dictionary representation of the value.
    """
    json_value = jsonable(value)
    if isinstance(json_value, Mapping):
        return dict(json_value)
    return {"value": json_value}


def _coerce_optional_mapping(value: Mapping[str, Any] | Any | None, default: Mapping[str, Any]) -> JsonRow:
    """Coerces an optional value into a JSON-compatible dictionary, using a default if None.

    Args:
        value: The value to coerce, which can be None.
        default: The default mapping to use if `value` is None.

    Returns:
        A dictionary representation of the value or the default mapping.
    """
    if value is None:
        return dict(default)
    return _coerce_mapping(value)


def _normalize_requests(source: Sequence[GenerationRequest | Mapping[str, Any]]) -> list[GenerationRequest]:
    """Normalizes a sequence of raw requests into a list of GenerationRequest objects.

    If an item is already a GenerationRequest, it's used directly. If it's a mapping,
    it's converted to a GenerationRequest, ensuring `sample_id` and `task_name` are set.

    Args:
        source: A sequence of GenerationRequest objects or mappings that can be
                converted to GenerationRequest.

    Returns:
        A list of normalized GenerationRequest objects.
    """
    requests: list[GenerationRequest] = []
    for index, item in enumerate(source):
        if isinstance(item, GenerationRequest):
            requests.append(item)
            continue
        row = dict(item)
        # Ensure sample_id and task_name are present for proper tracking
        row.setdefault("sample_id", f"sample-{index:04d}")
        row.setdefault("task_name", "contract_runner")
        requests.append(GenerationRequest.from_dict(row))
    return requests


def _is_failed(result: GenerationResult) -> bool:
    """Checks if a GenerationResult indicates a failure."""
    return not is_generation_result_successful(result)


def _coerce_generation_result(item: Any, sample_id: str) -> GenerationResult:
    """Coerces an item into a GenerationResult, ensuring it matches the given sample_id.

    If the item is already a GenerationResult but with a different sample_id,
    it creates a new one with the correct sample_id and adds the original sample_id to metadata.

    Args:
        item: The item to coerce (can be GenerationResult or a mapping).
        sample_id: The expected sample_id for the resulting GenerationResult.

    Returns:
        A GenerationResult object.
    """
    if isinstance(item, GenerationResult):
        if item.sample_id == sample_id:
            return item
        # If result is for a different sample_id, create a new one with correct ID, keeping original in metadata
        return GenerationResult(
            sample_id=sample_id,
            request_id=item.request_id,
            model_id=item.model_id,
            artifacts=item.artifacts,
            status=item.status,
            error=item.error,
            timings=item.timings,
            metadata={**dict(item.metadata), "source_sample_id": item.sample_id},
        )
    row = _coerce_mapping(item)
    row.setdefault("sample_id", sample_id)
    return GenerationResult.from_dict(row)


def _align_generation_results(requests: Sequence[GenerationRequest], raw_results: Sequence[Any]) -> list[GenerationResult]:
    """Aligns raw generation results with the original requests by sample_id.

    This handles cases where the runner returns results out of order or as a mix
    of keyed and sequential data. It ensures every request gets a corresponding
    GenerationResult, even if it's a "failed" placeholder.

    Args:
        requests: The original sequence of GenerationRequest objects.
        raw_results: The raw results returned by the WorldModelRunner.

    Returns:
        A list of GenerationResult objects, aligned with the input requests.
    """
    keyed: dict[str, Any] = {}
    sequential: list[Any] = []
    # Separate raw results into keyed (by sample_id) and sequential (no sample_id)
    for item in raw_results:
        sample_id = getattr(item, "sample_id", None)
        if sample_id is None and isinstance(item, Mapping):
            sample_id = item.get("sample_id")
        if sample_id is None:
            sequential.append(item)
        else:
            keyed[str(sample_id)] = item

    results: list[GenerationResult] = []
    sequential_index = 0
    # Iterate through requests to find or create corresponding results
    for request in requests:
        if request.sample_id in keyed:
            results.append(_coerce_generation_result(keyed[request.sample_id], request.sample_id))
        elif sequential_index < len(sequential):
            # Assign sequential results if no keyed match found
            results.append(_coerce_generation_result(sequential[sequential_index], request.sample_id))
            sequential_index += 1
        else:
            # If no result found for a request, create a failed placeholder
            results.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    status="failed",
                    error="runner did not return a result for sample",
                )
            )
    return results


def _index_rows_by_sample_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonRow]:
    """Indexes a sequence of rows (dictionaries) by their 'sample_id' field."""
    indexed: dict[str, JsonRow] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            indexed[str(sample_id)] = dict(row)
    return indexed


def _metric_results_from_row(row: Mapping[str, Any]) -> list[MetricResult]:
    """Extracts a list of MetricResult objects from a given row."""
    raw_results = row.get("metric_results")
    if not isinstance(raw_results, Iterable) or isinstance(raw_results, (str, bytes)):
        return []
    results: list[MetricResult] = []
    for item in raw_results:
        if isinstance(item, MetricResult):
            results.append(item)
        elif isinstance(item, Mapping):
            results.append(MetricResult.from_dict(item))
    return results


def _cached_metric_row_is_compatible(row: Mapping[str, Any], metrics: Sequence[Metric]) -> bool:
    """Checks if a cached metric row is compatible with the current set of metrics.

    Compatibility means:
    - If no metrics are defined, the cached row status must be 'not_run' and have no metric IDs.
    - If metrics are defined, the cached row status must be 'succeeded' and contain results
      for all and only the currently defined metric IDs.

    Args:
        row: The cached metric row (dictionary).
        metrics: The sequence of Metric objects for the current run.

    Returns:
        True if the cached row is compatible, False otherwise.
    """
    status = str(row.get("status") or "").lower()
    metric_ids = {metric_result.metric_id for metric_result in _metric_results_from_row(row)}
    if not metrics:
        return status == "not_run" and not metric_ids
    if status != "succeeded":
        return False
    return metric_ids == {_metric_id(metric) for metric in metrics}


@dataclass(frozen=True)
class _CachedSample:
    """Represents a successfully cached sample's data."""
    result: GenerationResult
    metric_row: JsonRow
    ledger_row: JsonRow


def _load_successful_sample_cache(
    paths: Mapping[str, Path],
    requests: Sequence[GenerationRequest],
    metrics: Sequence[Metric],
) -> dict[str, _CachedSample]:
    """Loads previously successful sample data from artifact files for resume functionality.

    This function reads existing `requests.jsonl`, `results.jsonl`, `per_sample.jsonl`,
    and `sample_ledger.jsonl` files. It then identifies samples that were previously
    successful and whose generated results and computed metrics are compatible with
    the current run's configuration (matching request, successful status, compatible metrics).

    Args:
        paths: A dictionary mapping artifact names to their file paths.
        requests: The sequence of GenerationRequest objects for the current run.
        metrics: The sequence of Metric objects for the current run.

    Returns:
        A dictionary where keys are sample IDs and values are `_CachedSample` objects,
        containing the GenerationResult, per-sample metric row, and sample ledger row
        for successfully cached and compatible samples.
    """
    # Read existing JSONL files and index their rows by sample_id
    request_rows = _index_rows_by_sample_id(_read_jsonl(paths["requests"]))
    result_rows = _index_rows_by_sample_id(_read_jsonl(paths["results"]))
    metric_rows = _index_rows_by_sample_id(_read_jsonl(paths["per_sample"]))
    ledger_rows = _index_rows_by_sample_id(_read_jsonl(paths["sample_ledger"]))

    cache: dict[str, _CachedSample] = {}
    for request in requests:
        sample_id = request.sample_id
        # Check if the request itself matches the cached request
        if request_rows.get(sample_id) != request.to_dict():
            continue
        
        # Ensure all necessary cached data exists for the sample
        result_row = result_rows.get(sample_id)
        metric_row = metric_rows.get(sample_id)
        ledger_row = ledger_rows.get(sample_id)
        if result_row is None or metric_row is None or ledger_row is None:
            continue
        
        # Check if the sample was marked as successful in the ledger
        if str(ledger_row.get("status") or "").lower() != "succeeded":
            continue
        
        # Check if the generation result itself was successful
        result = GenerationResult.from_dict(result_row)
        if _is_failed(result):
            continue
        
        # Check if the cached metrics are compatible with the current run's metrics configuration
        if not _cached_metric_row_is_compatible(metric_row, metrics):
            continue
        
        # If all checks pass, add the sample to the cache
        cache[sample_id] = _CachedSample(result=result, metric_row=metric_row, ledger_row=ledger_row)
    return cache


def _generate(runner: WorldModelRunner, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
    """Calls the runner's generate method and handles potential exceptions.

    If the runner's generation fails, it returns a list of failed GenerationResults
    for all requests with the error message.

    Args:
        runner: The WorldModelRunner instance.
        requests: The sequence of GenerationRequest objects.

    Returns:
        A list of GenerationResult objects, aligned with the input requests.
    """
    try:
        raw_results = runner.generate(requests)
    except Exception as exc:  # noqa: BLE001 - runner isolates model failures.
        # If the runner's generate method throws an exception,
        # return failed results for all requests
        return [
            GenerationResult(
                sample_id=request.sample_id,
                request_id=request.request_id,
                model_id=getattr(runner, "model_id", ""),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            for request in requests
        ]
    return _align_generation_results(requests, list(raw_results))


def _metric_id(metric: Metric) -> str:
    """Returns the ID of a metric, prioritizing its 'name' attribute, then class name."""
    return str(getattr(metric, "name", "") or metric.__class__.__name__)


def _metric_value(result: MetricResult) -> Any:
    """Returns the preferred value from a MetricResult (normalized_value if available, else raw_value)."""
    if result.normalized_value is not None:
        return result.normalized_value
    return result.raw_value


def _normalize_metric_output(output: Any, sample_id: str) -> list[MetricResult]:
    """Normalizes various forms of metric output into a list of MetricResult objects.

    Handles single MetricResult objects, dictionaries representing a single metric,
    dictionaries representing multiple metrics, and iterables of any of these.

    Args:
        output: The raw output from a metric's `compute_sample` method.
        sample_id: The sample ID associated with this metric output.

    Returns:
        A list of MetricResult objects.

    Raises:
        TypeError: If the output format is unsupported.
    """
    if output is None:
        return []
    if isinstance(output, MetricResult):
        if output.sample_id == sample_id:
            return [output]
        payload = output.to_dict()
        payload["sample_id"] = sample_id
        return [MetricResult.from_dict(payload)]
    if isinstance(output, Mapping):
        payload = dict(output)
        payload.setdefault("sample_id", sample_id)
        if "metric_id" in payload:
            # If it's a mapping with metric_id, assume it's a single MetricResult
            return [MetricResult.from_dict(payload)]
        # Otherwise, assume it's a mapping of metric_id -> value
        return [
            MetricResult(
                sample_id=sample_id,
                metric_id=str(metric_id),
                raw_value=value,
                # Attempt to normalize common numeric types to float
                normalized_value=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
            )
            for metric_id, value in payload.items()
            if metric_id not in {"sample_id", "status", "error"}  # Exclude control fields
        ]
    if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
        # Recursively process iterables
        results: list[MetricResult] = []
        for item in output:
            results.extend(_normalize_metric_output(item, sample_id))
        return results
    raise TypeError(f"unsupported metric output for sample {sample_id}: {type(output).__name__}")


def _compute_metrics_for_sample(
    request: GenerationRequest,
    result: GenerationResult,
    metrics: Sequence[Metric],
) -> tuple[JsonRow, list[MetricResult]]:
    """Compute one sample through the same capability-aware batch path."""

    return _compute_metrics_for_batch((request,), (result,), metrics)[0]


def _compute_metrics_for_batch(
    requests: Sequence[GenerationRequest],
    results: Sequence[GenerationResult],
    metrics: Sequence[Metric],
) -> list[tuple[JsonRow, list[MetricResult]]]:
    """Compute metric-major batches while preserving per-sample failure rows."""

    if len(requests) != len(results):
        raise ValueError(f"request/result batch sizes differ: {len(requests)} != {len(results)}")
    if not requests:
        return []

    collected: list[list[MetricResult]] = [[] for _ in requests]
    errors: list[list[dict[str, str]]] = [[] for _ in requests]
    eligible_indices = [index for index, result in enumerate(results) if not _is_failed(result)]

    for metric in metrics:
        metric_id = _metric_id(metric)
        compute_batch = getattr(metric, "compute_batch", None)
        batch_outputs: tuple[Any, ...] | None = None
        if callable(compute_batch) and eligible_indices:
            batch_requests = tuple(requests[index] for index in eligible_indices)
            batch_results = tuple(results[index] for index in eligible_indices)
            try:
                batch_outputs = align_batch_metric_outputs(
                    compute_batch(batch_requests, batch_results),
                    [request.sample_id for request in batch_requests],
                )
            except Exception:  # noqa: BLE001 - retain the sample-isolated fallback.
                batch_outputs = None

        for position, index in enumerate(eligible_indices):
            request = requests[index]
            result = results[index]
            try:
                output = (
                    batch_outputs[position]
                    if batch_outputs is not None
                    else metric.compute_sample(request, result)
                )
                collected[index].extend(_normalize_metric_output(output, request.sample_id))
            except Exception as exc:  # noqa: BLE001 - per-sample metric isolation is the contract.
                errors[index].append(
                    {"metric_id": metric_id, "message": f"{type(exc).__name__}: {exc}"}
                )

    rows: list[tuple[JsonRow, list[MetricResult]]] = []
    for index, (request, result) in enumerate(zip(requests, results)):
        if _is_failed(result):
            rows.append(
                (
                    {
                        "sample_id": request.sample_id,
                        "status": "skipped",
                        "metrics": {},
                        "error": result.error or "generation failed",
                    },
                    [],
                )
            )
            continue
        if not metrics:
            rows.append(({"sample_id": request.sample_id, "status": "not_run", "metrics": {}}, []))
            continue

        metric_results = collected[index]
        values = {
            metric_result.metric_id: _metric_value(metric_result)
            for metric_result in metric_results
            if metric_result.valid and _metric_value(metric_result) is not None
        }
        sample_errors = errors[index]
        row: JsonRow = {
            "sample_id": request.sample_id,
            "status": "failed" if sample_errors else "succeeded",
            "metrics": values,
            "metric_results": [metric_result.to_dict() for metric_result in metric_results],
        }
        if sample_errors:
            row["errors"] = sample_errors
            row["error"] = "; ".join(error["message"] for error in sample_errors)
        rows.append((row, metric_results))
    return rows


def _fallback_aggregate(metric_id: str, results: Sequence[MetricResult]) -> AggregateResult:
    """Provides a default aggregation (mean) if a metric doesn't define its own `aggregate` method.

    Args:
        metric_id: The identifier of the metric.
        results: The sequence of MetricResult objects for the metric across all samples.

    Returns:
        An AggregateResult object with basic mean statistics.
    """
    valid_values = [
        float(_metric_value(result))
        for result in results
        if result.valid and isinstance(_metric_value(result), (int, float)) and not isinstance(_metric_value(result), bool)
    ]
    stats = {"mean": sum(valid_values) / len(valid_values)} if valid_values else {}
    return AggregateResult(
        metric_id=metric_id,
        n_total=len(results),
        n_valid=len(valid_values),
        n_skipped=len(results) - len(valid_values),
        normalized_stats=stats,
        raw_stats=stats,
        valid=bool(valid_values),
    )


def _aggregate_metric(metric: Metric, results: Sequence[MetricResult]) -> AggregateResult:
    """Aggregates results for a single metric.

    If the metric provides an `aggregate` method, it's used. Otherwise, a fallback
    (mean calculation) is applied. Handles exceptions during aggregation.

    Args:
        metric: The Metric object.
        results: The sequence of MetricResult objects for this metric.

    Returns:
        An AggregateResult object.
    """
    metric_id = _metric_id(metric)
    try:
        aggregate = metric.aggregate(results)
    except Exception as exc:  # noqa: BLE001 - aggregate errors should be report data.
        # Capture exceptions during metric aggregation
        return AggregateResult(
            metric_id=metric_id,
            n_total=len(results),
            n_valid=0,
            n_skipped=len(results),
            valid=False,
            diagnostics={"error": f"{type(exc).__name__}: {exc}"},
        )
    if isinstance(aggregate, AggregateResult):
        return aggregate
    if isinstance(aggregate, Mapping):
        # Coerce mapping into AggregateResult
        payload = dict(aggregate)
        payload.setdefault("metric_id", metric_id)
        return AggregateResult.from_dict(payload)
    # Fallback to default aggregation if no custom method or unrecognized output
    return _fallback_aggregate(metric_id, results)


def _aggregate_value(aggregate: AggregateResult) -> float | int | None:
    """Extracts a primary numeric value (typically 'mean') from an AggregateResult for display."""
    for stats in (aggregate.normalized_stats, aggregate.raw_stats):
        value = stats.get("mean")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _build_summary(
    requests: Sequence[GenerationRequest],
    results: Sequence[GenerationResult],
    per_sample_rows: Sequence[Mapping[str, Any]],
    metric_results: Sequence[MetricResult],
    metrics: Sequence[Metric],
) -> dict[str, Any]:
    """Builds a comprehensive summary dictionary of the run's performance.

    This summary includes sample counts, success/failure breakdowns for generation
    and metrics, and detailed aggregated statistics for each metric.

    Args:
        requests: The sequence of original GenerationRequest objects.
        results: The sequence of GenerationResult objects.
        per_sample_rows: A sequence of dictionaries, each representing per-sample
                         metric computation results.
        metric_results: A flat list of all MetricResult objects from the run.
        metrics: The sequence of Metric objects used in the run.

    Returns:
        A dictionary containing the structured summary data.
    """
    sample_count = len(requests)
    
    # Identify failed generation samples
    generation_failed_ids = [result.sample_id for result in results if _is_failed(result)]
    
    # Identify failed or skipped metric computation samples
    metric_failed_ids = [
        str(row["sample_id"])
        for row in per_sample_rows
        if str(row.get("status") or "").lower() in {"failed", "failure", "error", "errored"}
    ]
    metric_skipped_ids = [
        str(row["sample_id"])
        for row in per_sample_rows
        if str(row.get("status") or "").lower() == "skipped"
    ]
    
    # Combine all failure/skipped IDs for total failed samples
    failed_ids = sorted(set(generation_failed_ids).union(metric_failed_ids).union(metric_skipped_ids))
    successful_samples = sample_count - len(failed_ids)

    # Group all individual metric results by their metric ID
    by_metric: dict[str, list[MetricResult]] = {}
    for metric_result in metric_results:
        by_metric.setdefault(metric_result.metric_id, []).append(metric_result)

    # Aggregate results for each defined metric
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, Any] = {}
    for metric in metrics:
        metric_id = _metric_id(metric)
        aggregate = _aggregate_metric(metric, by_metric.get(metric_id, ()))
        value = _aggregate_value(aggregate)
        per_metric[metric_id] = {
            "n_total": aggregate.n_total,
            "n_valid": aggregate.n_valid,
            "n_skipped": aggregate.n_skipped,
            "raw_stats": dict(aggregate.raw_stats),
            "normalized_stats": dict(aggregate.normalized_stats),
            "higher_is_better": getattr(metric, "higher_is_better", None),
            "valid": aggregate.valid,
            "diagnostics": dict(aggregate.diagnostics),
        }
        if aggregate.valid and value is not None:
            per_metric[metric_id]["mean"] = value
            leaderboard[metric_id] = value

    return {
        "schema_version": "worldfoundry-metrics-summary",
        "sample_count": sample_count,
        "successful_samples": successful_samples,
        "failed_samples": len(failed_ids),
        "failed_sample_ids": failed_ids,
        "generation": {
            "successful": sample_count - len(generation_failed_ids),
            "failed": len(generation_failed_ids),
            "failed_sample_ids": generation_failed_ids,
        },
        "metrics": {
            "enabled": bool(metrics),
            "successful": sum(1 for row in per_sample_rows if str(row.get("status") or "").lower() == "succeeded"),
            "failed": len(metric_failed_ids),
            "failed_sample_ids": metric_failed_ids,
            "skipped": len(metric_skipped_ids),
            "skipped_sample_ids": metric_skipped_ids,
        },
        "leaderboard": leaderboard,
        "per_metric": per_metric,
        "groups": {}, # Placeholder for future grouping functionality
    }


def _artifact_rows(results: Sequence[GenerationResult], output_dir: Path) -> list[JsonRow]:
    """Generates a list of artifact metadata rows for all artifacts across all results.

    Each row includes artifact details, enriched with its resolved URI relative to the output directory.

    Args:
        results: The sequence of GenerationResult objects.
        output_dir: The base output directory for resolving artifact paths.

    Returns:
        A list of dictionaries, each representing an artifact.
    """
    rows: list[JsonRow] = []
    for result in results:
        for name, artifact in result.artifacts.items():
            # Enrich artifact reference with resolved path relative to output_dir
            row = enrich_artifact_ref(artifact, base_dir=output_dir).to_dict()
            row.setdefault("sample_id", result.sample_id)
            row.setdefault("name", name)
            rows.append(row)
    return rows


def _artifact_paths(output_dir: Path, artifact_count: int) -> dict[str, str]:
    """Generates a dictionary of standard artifact file paths for the run report.

    Args:
        output_dir: The base output directory for the run.
        artifact_count: The total number of artifacts generated. If 0, 'artifacts.jsonl' is omitted.

    Returns:
        A dictionary mapping artifact logical names to their resolved absolute string paths.
    """
    paths = {
        "run_manifest": output_dir / "run_manifest.json",
        "environment": output_dir / "environment.json",
        "env_requirements": output_dir / "env_requirements.json",
        "execution_plan": output_dir / "execution_plan.json",
        "requests": output_dir / "requests.jsonl",
        "results": output_dir / "results.jsonl",
        "sample_ledger": output_dir / "sample_ledger.jsonl",
        "per_sample_metrics": output_dir / "metrics" / "per_sample.jsonl",
        "summary": output_dir / "metrics" / "summary.json",
        "run_summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "scorecard": output_dir / "scorecard.json",
    }
    if artifact_count:
        # Only include artifacts.jsonl if there are actual artifacts
        paths["artifacts"] = output_dir / "artifacts.jsonl"
    # Resolve all paths to absolute strings
    return {name: str(path.resolve()) for name, path in paths.items()}


def _coerce_request(
    request: ContractRunRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> ContractRunRequest:
    """Coerces various input types into a ContractRunRequest object.

    Handles `ContractRunRequest` objects (potentially updated with kwargs),
    and mappings (dictionaries) that define the request. Ensures required
    fields (`output_dir`, `requests`, `runner`) are present.

    Args:
        request: The primary request object or mapping.
        kwargs: Additional keyword arguments to potentially override/add to the request.

    Returns:
        A fully formed ContractRunRequest object.

    Raises:
        TypeError: If essential fields are missing after coercion.
    """
    if isinstance(request, ContractRunRequest):
        if kwargs:
            # If kwargs are provided, merge them with the existing request
            payload = asdict(request)
            payload.update(kwargs)
            return ContractRunRequest(**payload)
        return request

    # Start with kwargs, then overlay the request mapping if it exists
    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}
    
    # Validate required fields
    if "output_dir" not in payload:
        raise TypeError("run_contract requires an output_dir")
    if "requests" not in payload:
        raise TypeError("run_contract requires requests")
    if "runner" not in payload:
        raise TypeError("run_contract requires a runner")
    
    return ContractRunRequest(**payload)


def run_contract(
    request: ContractRunRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ContractRunResult:
    """Core execution engine for processing batches through a local model runner.

    Workflow:
    1. Prepares execution plan schemas and writes an initial running manifest.
    2. Leverages `run_generation_with_cache` to dispatch workload securely to the `runner`, 
       yielding an interleaved sequence of cached hit results and newly computed results.
    3. Runs the newly generated results against defined evaluators/metrics.
    4. Aggregates results, serializes scorecards and summary JSONL data, and gracefully cleans up resources.

    Args:
        request: A `ContractRunRequest` object or a dictionary to configure the run.
        **kwargs: Additional keyword arguments to override or add to the request.

    Returns:
        A `ContractRunResult` object summarizing the outcome of the run.
    """
    # Coerce the input request into a standardized ContractRunRequest object
    run_request = _coerce_request(request, kwargs)
    
    # Initialize output directories and generate a unique run ID
    output_dir = Path(run_request.output_dir).resolve()
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_request.run_id or f"contract-{uuid4().hex[:12]}"
    started_at = _utcnow_iso()
    
    # Normalize input requests into a list of GenerationRequest objects
    requests = _normalize_requests(run_request.requests)
    
    # Define all standard output file paths
    paths = {
        "manifest": output_dir / "run_manifest.json",
        "environment": output_dir / "environment.json",
        "env_requirements": output_dir / "env_requirements.json",
        "execution_plan": output_dir / "execution_plan.json",
        "requests": output_dir / "requests.jsonl",
        "results": output_dir / "results.jsonl",
        "artifacts": output_dir / "artifacts.jsonl",
        "sample_ledger": output_dir / "sample_ledger.jsonl",
        "per_sample": metrics_dir / "per_sample.jsonl",
        "summary": metrics_dir / "summary.json",
        "run_summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "scorecard": output_dir / "scorecard.json",
    }
    
    # Load previously successful samples from disk if resume is enabled
    successful_sample_cache = _load_successful_sample_cache(paths, requests, run_request.metrics) if run_request.resume else {}
    
    # Reset JSONL files to ensure a clean start for non-resumed samples
    for key in ("requests", "results", "artifacts", "sample_ledger", "per_sample"):
        reset_jsonl(paths[key])

    runner = run_request.runner
    model_id = str(getattr(runner, "model_id", "contract-model"))
    
    # Coerce and set default values for benchmark, model, and dataset metadata
    benchmark = _coerce_optional_mapping(
        run_request.benchmark,
        {
            "suite": "contract_runner",
            "benchmark_name": "contract_runner",
            "task_type": "contract_runner",
            "evaluation_protocol": "contract_runner",
        },
    )
    model = _coerce_optional_mapping(run_request.model, {"model_type": "contract", "model_name": model_id})
    dataset = _coerce_optional_mapping(run_request.dataset, {"split": "contract"})
    dataset.setdefault("sample_count", len(requests))
    
    # Build version context and run fingerprint for reproducibility
    version_context = build_version_context(
        runner="contract_runner",
        benchmark=benchmark,
        model=model,
        dataset=dataset,
        model_runner=runner,
        metrics=run_request.metrics,
    )
    run_fingerprint = build_run_fingerprint(
        version_context=version_context,
        requests=requests,
    )

    # Create and write the execution plan
    plan = {
        "schema_version": "worldfoundry-execution-plan",
        "run_id": run_id,
        "created_at": started_at,
        "runner": "contract_runner",
        "version_context": version_context,
        "run_fingerprint": run_fingerprint,
        "stages": ["materialize", "load_model", "generate", "evaluate", "aggregate", "report"],
        "sample_count": len(requests),
        "samples": [
            {"index": index, "sample_id": request_row.sample_id, "status": "planned"}
            for index, request_row in enumerate(requests)
        ],
        "outputs": {
            "requests": "requests.jsonl",
            "results": "results.jsonl",
            "artifacts": "artifacts.jsonl",
            "sample_ledger": "sample_ledger.jsonl",
            "per_sample_metrics": "metrics/per_sample.jsonl",
            "summary": "metrics/summary.json",
            "run_summary": "summary.json",
            "report": "report.md",
            "scorecard": "scorecard.json",
        },
    }
    write_json(paths["execution_plan"], plan)

    # Write the initial run manifest with 'running' status
    initial_manifest = {
        "schema_version": "worldfoundry-run-manifest",
        "run_id": run_id,
        "runner": "contract_runner",
        "status": "running",
        "started_at": started_at,
        "output_dir": str(output_dir),
        "benchmark": benchmark,
        "model": model,
        "dataset": dataset,
        "version_context": version_context,
        "run_fingerprint": run_fingerprint,
        "sample_count": len(requests),
        "execution_plan": str(paths["execution_plan"].resolve()),
    }
    write_run_manifest_artifacts(
        output_dir=output_dir,
        base_manifest=initial_manifest,
        config={
            "runner": "contract_runner",
            "fail_on_sample_error": run_request.fail_on_sample_error,
            "write_artifacts_index": run_request.write_artifacts_index,
            "cleanup_runner": run_request.cleanup_runner,
            "resume": run_request.resume,
            "generation_cache_mode": run_request.generation_cache_mode,
            "generation_cache_namespace": run_request.generation_cache_namespace,
        },
        cache_paths=cache_paths_from_stats({}), # No cache stats yet at this point
        package_names=("worldfoundry", "numpy", "pandas"),
        manifest_path=paths["manifest"],
        environment_path=paths["environment"],
        env_requirements_path=paths["env_requirements"],
    )

    # Write immutable stage outputs with one CPFS open/close cycle each.
    write_jsonl(paths["requests"], (request_row.to_dict() for request_row in requests), atomic=False)

    # Filter out requests that were successfully loaded from cache (for resume)
    pending_requests = [
        request_row for request_row in requests if request_row.sample_id not in successful_sample_cache
    ]
    try:
        # Run generation for pending requests, utilizing the generation cache
        generated_results, generation_cache_stats = run_generation_with_cache(
            pending_requests,
            lambda rows: _generate(runner, rows),
            cache_dir=run_request.generation_cache_dir,
            cache_mode=run_request.generation_cache_mode,
            namespace=run_request.generation_cache_namespace,
            version_context=version_context,
            artifact_base_dir=output_dir,
            run_id=run_id,
        )
    finally:
        # Ensure runner cleanup is attempted even if generation fails
        if run_request.cleanup_runner:
            cleanup = getattr(runner, "cleanup", None)
            if callable(cleanup):
                cleanup()

    # Consolidate generated results with cached results
    generated_by_sample_id = {
        result.sample_id: result
        for result in generated_results
    }
    results = [
        successful_sample_cache[request_row.sample_id].result  # Use cached result if available
        if request_row.sample_id in successful_sample_cache
        else generated_by_sample_id[request_row.sample_id]  # Otherwise, use newly generated result
        for request_row in requests
    ]
    
    # Write all final generation results to results.jsonl.
    write_jsonl(paths["results"], (result.to_dict() for result in results), atomic=False)

    # Process and write artifact metadata if enabled
    artifact_rows = _artifact_rows(results, output_dir)
    if artifact_rows and run_request.write_artifacts_index:
        write_jsonl(paths["artifacts"], artifact_rows, atomic=False)

    per_sample_rows: list[JsonRow] = []
    all_metric_results: list[MetricResult] = []
    metric_indices = [
        index
        for index, request_row in enumerate(requests)
        if request_row.sample_id not in successful_sample_cache
    ]
    computed_metric_outputs = dict(
        zip(
            metric_indices,
            _compute_metrics_for_batch(
                [requests[index] for index in metric_indices],
                [results[index] for index in metric_indices],
                run_request.metrics,
            ),
        )
    )

    # Iterate through each sample while keeping the progress streams open.
    with (
        JsonlWriter(paths["sample_ledger"]) as ledger_writer,
        JsonlWriter(paths["per_sample"]) as per_sample_writer,
    ):
        for index, request_row in enumerate(requests):
            result = results[index]
            generation_status = "failed" if _is_failed(result) else "succeeded"
            cached_sample = successful_sample_cache.get(request_row.sample_id)
            generation_cache_metadata = generation_cache_hit_metadata(result)

            # Determine if metrics should be loaded from cache or recomputed.
            if cached_sample is not None:
                metric_row = dict(cached_sample.metric_row)
                metric_results = _metric_results_from_row(metric_row)
            else:
                metric_row, metric_results = computed_metric_outputs[index]

            metric_status = str(metric_row.get("status") or "succeeded").lower()
            per_sample_rows.append(metric_row)
            all_metric_results.extend(metric_results)

            # Collect errors for the sample ledger.
            errors = []
            if generation_status == "failed":
                errors.append({"stage": "generate", "message": result.error or "generation failed"})
            if metric_status in {"failed", "failure", "error", "errored", "skipped"}:
                errors.append({"stage": "evaluate", "message": str(metric_row.get("error") or "metric failed")})

            ledger_writer.write(
                {
                    "run_id": run_id,
                    "sample_id": request_row.sample_id,
                    "index": index,
                    "status": "failed" if errors else "succeeded",
                    "generation_status": generation_status,
                    "metrics_status": metric_status,
                    "started_at": started_at,
                    "finished_at": _utcnow_iso(),
                    # Add cache hit metadata if applicable (resume or generation cache).
                    **(
                        {
                            "cached": True,
                            "cache_source": "output_dir_resume",
                            "source_run_id": cached_sample.ledger_row.get("run_id"),
                        }
                        if cached_sample is not None
                        else {}
                    ),
                    **(
                        {
                            "cached": True,
                            "cache_source": "generation_result_cache",
                            "source_run_id": generation_cache_metadata.get("source_run_id"),
                            "cache_key_hash": generation_cache_metadata.get("key_hash"),
                        }
                        if generation_cache_metadata
                        else {}
                    ),
                    **({"errors": errors} if errors else {}),
                }
            )
            per_sample_writer.write(metric_row)

    # Compile the final statistics and summaries
    summary = _build_summary(requests, results, per_sample_rows, all_metric_results, run_request.metrics)
    write_json(paths["summary"], summary)

    # Derive execution exit-codes based on strict fault-tolerance configurations
    failed_sample_count = int(summary["failed_samples"])
    status = "completed_with_failures" if failed_sample_count else "succeeded"
    exit_code = 1 if failed_sample_count and run_request.fail_on_sample_error else 0
    finished_at = _utcnow_iso()
    artifact_count = len(artifact_rows) if run_request.write_artifacts_index else 0
    artifact_paths = _artifact_paths(output_dir, artifact_count)
    
    # Prepare data for the scorecard
    generation = {
        "num_requests": len(requests),
        "successful": int(summary["generation"]["successful"]),
        "failed": int(summary["generation"]["failed"]),
        "error_sample_ids": list(summary["generation"]["failed_sample_ids"]),
        "throughput": {}, # Placeholder for future throughput stats
    }
    skipped = {
        "count": int(summary["metrics"]["skipped"]),
        "sample_ids": list(summary["metrics"]["skipped_sample_ids"]),
    }
    
    # Write the canonical standard WorldFoundry Scorecard reporting payload
    write_scorecard(
        paths["scorecard"],
        run={
            "run_id": run_id,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "version_context": version_context,
            "run_fingerprint": run_fingerprint,
            "generation_cache": generation_cache_stats.to_dict(),
        },
        benchmark=benchmark,
        model=model,
        dataset=dataset,
        generation=generation,
        metrics_summary=summary,
        artifacts=artifact_paths,
        skipped=skipped,
        evaluation_kind="contract_runner",
    )
    
    # Write the summary JSON and markdown report
    write_run_report_artifacts(
        output_dir=output_dir,
        scorecard_path=paths["scorecard"],
        summary_path=paths["run_summary"],
        report_path=paths["report"],
    )

    # Package the final run manifest with updated status and results
    final_manifest = dict(initial_manifest)
    final_manifest.update(
        {
            "status": status,
            "finished_at": finished_at,
            "exit_code": exit_code,
            "successful_sample_count": int(summary["successful_samples"]),
            "failed_sample_count": failed_sample_count,
            "generation_cache": generation_cache_stats.to_dict(),
            "artifacts": artifact_paths,
        }
    )
    write_run_manifest_artifacts(
        output_dir=output_dir,
        base_manifest=final_manifest,
        config={
            "runner": "contract_runner",
            "fail_on_sample_error": run_request.fail_on_sample_error,
            "write_artifacts_index": run_request.write_artifacts_index,
            "cleanup_runner": run_request.cleanup_runner,
            "resume": run_request.resume,
            "generation_cache_mode": run_request.generation_cache_mode,
            "generation_cache_namespace": run_request.generation_cache_namespace,
        },
        cache_paths=cache_paths_from_stats(generation_cache_stats), # Include final cache stats
        package_names=("worldfoundry", "numpy", "pandas"),
        manifest_path=paths["manifest"],
        environment_path=paths["environment"],
        env_requirements_path=paths["env_requirements"],
    )

    # Return the final ContractRunResult
    return ContractRunResult(
        status=status,
        exit_code=exit_code,
        output_dir=output_dir,
        manifest_path=paths["manifest"].resolve(),
        execution_plan_path=paths["execution_plan"].resolve(),
        scorecard_path=paths["scorecard"].resolve(),
        sample_count=len(requests),
        successful_sample_count=int(summary["successful_samples"]),
        failed_sample_count=failed_sample_count,
        artifact_count=artifact_count,
    )


execute_contract_run = run_contract


__all__ = [
    "ContractRunRequest",
    "ContractRunResult",
    "ContractRunner",
    "execute_contract_run",
    "run_contract",
]
