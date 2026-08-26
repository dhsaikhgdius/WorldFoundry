"""CLI sub-commands for dataset manifest creation, inspection, validation, and materialization.

Provides the ``dataset`` sub-command under ``worldfoundry-eval`` with
``create``, ``show``, ``validate``, ``materialize``, plus DatasetManager
helpers ``locate`` and ``plan`` (DS-09).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worldfoundry.cli.utils import json_dump, parse_key_value_mapping
from worldfoundry.evaluation.utils import write_json, write_jsonl


def _handle_dataset_create(args: argparse.Namespace) -> int:
    """Create a dataset manifest from JSON or JSONL samples.

    Args:
        args: Parsed `dataset create` CLI arguments.
    """
    from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest

    manifest = build_dataset_manifest(
        samples_path=args.samples_path,
        dataset_id=args.dataset_id,
        split=args.split,
        root=args.root,
        source_uri=args.source_uri,
        license=args.license,
        access=parse_key_value_mapping(args.access),
        metadata=parse_key_value_mapping(args.metadata),
    )
    if args.output_json:
        write_dataset_manifest(manifest, args.output_json)
    payload = manifest.to_dict()
    if args.json:
        json_dump(payload)
    else:
        print(
            f"dataset_manifest: {payload['dataset_id']} "
            f"samples={payload['sample_count']} sha256={payload['sha256']}"
        )
        if args.output_json:
            print(f"wrote: {args.output_json}")
    return 0


def _handle_dataset_show(args: argparse.Namespace) -> int:
    """Print one dataset manifest.

    Args:
        args: Parsed `dataset show` CLI arguments.
    """
    from worldfoundry.evaluation.tasks.datasets import load_dataset_manifest

    manifest = load_dataset_manifest(args.manifest)
    payload = manifest.to_dict()
    if args.json:
        json_dump(payload)
        return 0
    print(f"schema_version: {payload['schema_version']}")
    print(f"dataset_id: {payload['dataset_id']}")
    print(f"split: {payload['split']}")
    print(f"samples_path: {payload['samples_path']}")
    print(f"sample_count: {payload['sample_count']}")
    print(f"sha256: {payload['sha256']}")
    return 0


def _handle_dataset_validate(args: argparse.Namespace) -> int:
    """Validate a dataset manifest against its sample file.

    Args:
        args: Parsed `dataset validate` CLI arguments.
    """
    from worldfoundry.evaluation.tasks.datasets import validate_dataset_manifest

    payload = validate_dataset_manifest(args.manifest)
    if args.json:
        json_dump(payload)
    else:
        print(f"ok: {payload['ok']}")
        print(f"dataset_id: {payload.get('dataset_id') or '-'}")
        print(f"samples_path: {payload.get('samples_path') or '-'}")
        print(f"sample_count: {payload.get('sample_count')}")
        for warning in payload.get("warnings", ()):
            print(f"warning: {warning}")
        for issue in payload.get("issues", ()):
            print(f"issue: {issue}")
    return 0 if payload["ok"] else 1


def _handle_dataset_materialize(args: argparse.Namespace) -> int:
    """Materialize generation request rows from a dataset manifest.

    Args:
        args: Parsed `dataset materialize` CLI arguments.
    """
    from worldfoundry.evaluation.runner import materialize_requests_from_dataset_manifest

    materialized = materialize_requests_from_dataset_manifest(
        args.manifest,
        task_name=args.task_name,
        split=args.split,
        input_keys=tuple(args.input_key or ()),
        output_keys=tuple(args.output_key or ("generated_video",)),
        limit=args.num_samples,
    )
    payload = materialized.to_dict()
    if args.output_json:
        write_json(args.output_json, payload, atomic=False)
    if args.output_jsonl:
        write_jsonl(
            args.output_jsonl,
            [request.to_dict() for request in materialized.requests],
            atomic=False,
        )
    if args.json:
        json_dump(payload)
    else:
        print(
            f"materialized: task={materialized.task_type} "
            f"dataset={materialized.benchmark_name} requests={materialized.sample_count}"
        )
        if args.output_json:
            print(f"wrote_json: {args.output_json}")
        if args.output_jsonl:
            print(f"wrote_jsonl: {args.output_jsonl}")
    return 0


def _dataset_manager(args: argparse.Namespace):
    from worldfoundry.evaluation.tasks.datasets import DatasetManager
    from worldfoundry.runtime.env import resolve_hf_cache_dir

    cache_dir = args.cache_dir or resolve_hf_cache_dir()
    return DatasetManager(cache_dir)


def _handle_dataset_locate(args: argparse.Namespace) -> int:
    """Locate a local Hugging Face dataset via DatasetManager (DS-09)."""

    manager = _dataset_manager(args)
    location = manager.locate_local(
        args.dataset_id,
        data_root=args.data_root,
        manifest_path=args.manifest,
    )
    payload = location.to_dict()
    if args.json:
        json_dump(payload)
    else:
        print(f"ok: {payload.get('ok')}")
        print(f"hf_dataset_id: {payload.get('hf_dataset_id') or '-'}")
        print(f"source: {payload.get('source')}")
        print(f"status: {payload.get('status')}")
        print(f"path: {payload.get('path') or '-'}")
        if payload.get("reason"):
            print(f"reason: {payload['reason']}")
    return 0 if location.ready else 1


def _handle_dataset_plan(args: argparse.Namespace) -> int:
    """Print a DatasetManager download plan for one or more dataset ids (DS-09)."""

    manager = _dataset_manager(args)
    plan = manager.build_download_plan(
        list(args.dataset_id),
        dataset_ids=args.dataset_id,
        check_local=bool(args.check_local),
    )
    payload = plan.to_dict()
    if args.json:
        json_dump(payload)
    else:
        print(f"commands: {len(payload.get('commands') or ())}")
        for command in payload.get("commands") or ():
            if isinstance(command, (list, tuple)):
                print("  " + " ".join(str(part) for part in command))
            else:
                print(f"  {command}")
        for check in payload.get("local_checks") or ():
            if isinstance(check, dict):
                print(
                    f"local: {check.get('hf_dataset_id') or '-'} "
                    f"status={check.get('status')} ready={check.get('ready')}"
                )
    return 0


def register_dataset_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register dataset CLI commands.

    Args:
        subparsers: Root argparse subparser collection.
    """
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Create, inspect, validate, materialize manifests; locate/plan HF datasets",
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    dataset_create_parser = dataset_subparsers.add_parser(
        "create",
        help="Create a worldfoundry-dataset-manifest JSON file from a samples JSON/JSONL file",
    )
    dataset_create_parser.add_argument("--samples-path", type=Path, required=True)
    dataset_create_parser.add_argument("--output-json", type=Path)
    dataset_create_parser.add_argument("--dataset-id")
    dataset_create_parser.add_argument("--split", default="default")
    dataset_create_parser.add_argument("--root", type=Path)
    dataset_create_parser.add_argument("--source-uri")
    dataset_create_parser.add_argument("--license")
    dataset_create_parser.add_argument("--access", action="append", default=None, metavar="KEY=VALUE")
    dataset_create_parser.add_argument("--metadata", action="append", default=None, metavar="KEY=VALUE")
    dataset_create_parser.add_argument("--json", action="store_true")
    dataset_create_parser.set_defaults(func=_handle_dataset_create)

    dataset_show_parser = dataset_subparsers.add_parser("show", help="Show one dataset manifest")
    dataset_show_parser.add_argument("manifest", type=Path)
    dataset_show_parser.add_argument("--json", action="store_true")
    dataset_show_parser.set_defaults(func=_handle_dataset_show)

    dataset_validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="Validate a dataset manifest against its samples file",
    )
    dataset_validate_parser.add_argument("manifest", type=Path)
    dataset_validate_parser.add_argument("--json", action="store_true")
    dataset_validate_parser.set_defaults(func=_handle_dataset_validate)

    dataset_materialize_parser = dataset_subparsers.add_parser(
        "materialize",
        help="Materialize GenerationRequest rows from a dataset manifest",
    )
    dataset_materialize_parser.add_argument("manifest", type=Path)
    dataset_materialize_parser.add_argument("--task-name", required=True)
    dataset_materialize_parser.add_argument("--split")
    dataset_materialize_parser.add_argument("--input-key", action="append", default=None)
    dataset_materialize_parser.add_argument("--output-key", action="append", default=None)
    dataset_materialize_parser.add_argument("--num-samples", type=int)
    dataset_materialize_parser.add_argument("--output-json", type=Path)
    dataset_materialize_parser.add_argument("--output-jsonl", type=Path)
    dataset_materialize_parser.add_argument("--json", action="store_true")
    dataset_materialize_parser.set_defaults(func=_handle_dataset_materialize)

    dataset_locate_parser = dataset_subparsers.add_parser(
        "locate",
        help="Locate a local Hugging Face dataset (env / manifest / direct root / HF cache)",
    )
    dataset_locate_parser.add_argument("dataset_id", help="Hugging Face dataset id, e.g. org/name")
    dataset_locate_parser.add_argument("--data-root", type=Path)
    dataset_locate_parser.add_argument("--manifest", type=Path, help="Optional local asset / data manifest")
    dataset_locate_parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache root override")
    dataset_locate_parser.add_argument("--json", action="store_true")
    dataset_locate_parser.set_defaults(func=_handle_dataset_locate)

    dataset_plan_parser = dataset_subparsers.add_parser(
        "plan",
        help="Build DatasetManager download commands for one or more HF dataset ids",
    )
    dataset_plan_parser.add_argument(
        "dataset_id",
        nargs="+",
        help="Hugging Face dataset id(s), e.g. org/name",
    )
    dataset_plan_parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache root override")
    dataset_plan_parser.add_argument(
        "--check-local",
        action="store_true",
        help="Include local readiness checks in the plan payload",
    )
    dataset_plan_parser.add_argument("--json", action="store_true")
    dataset_plan_parser.set_defaults(func=_handle_dataset_plan)
