#!/usr/bin/env python3
"""Run model/generated videos through every integrated video benchmark runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from typing import Any

from worldfoundry.core.io.paths import project_root


REPO_ROOT = project_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.evaluation.api import GENERATION_REQUEST_SCHEMA_VERSION  # noqa: E402
from worldfoundry.evaluation.runner import ModelBenchmarkSuiteRequest, run_model_benchmark_suite  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.benchmark_data import (  # noqa: E402
    materialize_sample_generated_videos,
)
from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY  # noqa: E402
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR  # noqa: E402

DEFAULT_PROMPT = "A small robot explores a bright workshop, cinematic motion, high quality."


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_key_value(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or ():
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        parsed[key.strip()] = _json_value(value)
    return parsed


def _parse_ids(raw_ids: list[str] | None, *, known: set[str] | None = None, field: str = "id") -> list[str]:
    parsed: list[str] = []
    for item in raw_ids or ():
        parsed.extend(part.strip() for part in item.split(",") if part.strip())
    parsed = list(dict.fromkeys(parsed))
    if known is not None:
        unknown = sorted(set(parsed) - known)
        if unknown:
            raise SystemExit(f"unknown {field}(s): {', '.join(unknown)}")
    return parsed


def _write_prompt_requests(output_dir: Path, prompts: list[str], *, output_artifact: str) -> Path:
    requests_path = output_dir / "requests.jsonl"
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, prompt in enumerate(prompts):
        rows.append(
            {
                "schema_version": GENERATION_REQUEST_SCHEMA_VERSION,
                "sample_id": f"sample-{index:04d}",
                "task_name": "video_generation",
                "split": "validation",
                "inputs": {"prompt": prompt},
                "output_schema": {output_artifact: {"kind": output_artifact}},
            }
        )
    requests_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return requests_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more WorldFoundry video models, or an existing generated-video directory, "
            "through all registered video benchmark runners."
        )
    )
    parser.add_argument("--model", action="append", dest="model_ids", default=None, help="Model id or alias. Repeat or comma-separate.")
    parser.add_argument("--benchmark-id", action="append", default=None, help="Video benchmark id. Defaults to all 35 registered video benches.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / "video_model_benchmark_suite")
    parser.add_argument("--mode", choices=("contract", "official-validation", "official-run"), default="official-validation")
    parser.add_argument("--generated-artifact-dir", type=Path, help="Existing model video directory; skips generation.")
    parser.add_argument("--requests-path", type=Path, help="GenerationRequest JSON/JSONL file. Auto-created from --prompt when omitted.")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt for auto-created generation requests. Repeat for multiple samples.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--model-manifest-dir", type=Path, default=MODEL_ZOO_DIR)
    parser.add_argument("--model-runner", help="Override model runner target.")
    parser.add_argument("--model-variant", help="Model-zoo variant id.")
    parser.add_argument("--model-parameter", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--model-runtime", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--output-artifact", default="generated_video")
    parser.add_argument("--required-artifact", action="append", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--model-workers", type=int, default=1)
    parser.add_argument(
        "--model-worker-device",
        action="append",
        default=None,
        metavar="CUDA_VISIBLE_DEVICES",
        help="CUDA device group for one model worker; repeat for multiple workers.",
    )
    parser.add_argument("--no-skip-incompatible", dest="skip_incompatible", action="store_false", default=True)
    parser.add_argument(
        "--fixture-video",
        action="store_true",
        help="Use generated sample videos instead of running a model.",
    )
    parser.add_argument("--video-count", type=int, default=2, help="Fixture video count.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_ids = _parse_ids(args.benchmark_id, known=set(VIDEO_RUNNER_REGISTRY), field="benchmark id")
    if not benchmark_ids:
        benchmark_ids = sorted(VIDEO_RUNNER_REGISTRY)

    model_ids = _parse_ids(args.model_ids, field="model id")
    generated_artifact_dir = args.generated_artifact_dir
    requests_path = args.requests_path

    use_fixture = args.fixture_video
    if use_fixture:
        generated_artifact_dir = materialize_sample_generated_videos(output_dir, count=max(1, args.video_count))
        model_ids = model_ids or ["tiny-video-fixture"]
    elif generated_artifact_dir is not None:
        model_ids = model_ids or ["external-generated-videos"]
    elif requests_path is None:
        if not model_ids:
            raise SystemExit("provide --model, --generated-artifact-dir, or explicit --fixture-video")
        prompts = args.prompt or [DEFAULT_PROMPT]
        requests_path = _write_prompt_requests(output_dir, prompts, output_artifact=args.output_artifact)

    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=output_dir,
            benchmark_manifest_dir=BENCHMARK_ZOO_DIR,
            model_manifest_dir=args.model_manifest_dir,
            model_ids=tuple(model_ids),
            benchmark_ids=tuple(benchmark_ids),
            mode=args.mode,
            execute=not args.plan_only,
            model_workers=args.model_workers,
            worker_cuda_devices=tuple(args.model_worker_device or ()),
            skip_incompatible=args.skip_incompatible,
            model_runner=args.model_runner,
            model_variant_id=args.model_variant,
            model_parameters=_parse_key_value(args.model_parameter),
            model_runtime=_parse_key_value(args.model_runtime),
            requests_path=requests_path,
            num_samples=args.num_samples,
            generated_artifact_dir=generated_artifact_dir,
            output_artifact=args.output_artifact,
            required_artifacts=tuple(args.required_artifact) if args.required_artifact is not None else (args.output_artifact,),
            benchmark_timeout_seconds=args.timeout_seconds,
            resume=args.resume,
        )
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {})
        print(
            f"video model benchmark suite {payload['status']}: "
            f"{summary.get('succeeded', 0)}/{summary.get('total', 0)} succeeded, "
            f"{summary.get('failed', 0)} failed, {summary.get('skipped', 0)} skipped"
        )
        print(f"suite manifest: {payload['suite_manifest_path']}")
        print(f"report: {payload['suite_report_path']}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
