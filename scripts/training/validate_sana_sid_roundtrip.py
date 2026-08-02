#!/usr/bin/env python3
"""Run a real-weight SANA SiD update, exact resume, and export-reload gate."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.core.io.integrity import replace_json_atomic
from worldfoundry.training.engine.sana.sid import materialize_sana_sid_training_run
from worldfoundry.training.models.sana_sid import build_local_diffusers_sana_sid_adapter
from worldfoundry.training.recipes import PostTrainingRecipe
from worldfoundry.training.tuning import FullModelArtifact, load_full_model

_LOADER_DIAGNOSTIC_FIELDS = ("_num_yielded", "_sampler_iter_yielded")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/post_training/sana_sprint_600m_sid.yaml"),
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    return parser.parse_args()


def _gate_recipe(
    source: PostTrainingRecipe,
    *,
    manifest: Path,
    cache: Path,
    work_dir: Path,
    steps: int,
    height: int | None,
    width: int | None,
) -> PostTrainingRecipe:
    payload = source.to_dict()
    payload["run"] = {
        "id": "sana-sprint-600m-sid-real-gate",
        "output_dir": str(work_dir / "recipe-owned-run"),
    }
    data = dict(payload["data"])
    options = dict(data["options"])
    options.update(
        {
            "microbatch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "snapshot_every_n_steps": 1,
        }
    )
    options.pop("prefetch_factor", None)
    if height is not None:
        options["height"] = int(height)
    if width is not None:
        options["width"] = int(width)
    data.update(
        {
            "manifest": str(manifest),
            "cache": str(cache),
            "shuffle": False,
            "tail_policy": "drop",
            "options": options,
        }
    )
    payload["data"] = data
    payload["distributed"] = {"backend": "single"}
    payload["checkpoint"] = {
        "save_every_steps": int(steps),
        "async": False,
        "export_every_steps": 0,
    }
    payload["export"] = {
        "format": "safetensors",
        "options": dict(source.export.options),
    }
    return PostTrainingRecipe.from_mapping(payload)


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _framed(hasher: object, label: str, payload: bytes) -> None:
    hasher.update(label.encode("utf-8"))
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _update_digest(hasher: object, value: object) -> None:
    if value is None:
        _framed(hasher, "none", b"")
        return
    if isinstance(value, bool):
        _framed(hasher, "bool", b"1" if value else b"0")
        return
    if isinstance(value, int):
        _framed(hasher, "int", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        _framed(hasher, "float", value.hex().encode("ascii"))
        return
    if isinstance(value, str):
        _framed(hasher, "str", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _framed(hasher, "bytes", value)
        return
    if isinstance(value, Path):
        _framed(hasher, "path", str(value).encode("utf-8"))
        return
    if isinstance(value, (torch.dtype, torch.device)):
        _framed(hasher, type(value).__name__, str(value).encode("ascii"))
        return
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            raise TypeError(f"gate digest does not support tensor layout {value.layout}")
        _framed(hasher, "tensor-dtype", str(value.dtype).encode("ascii"))
        _update_digest(hasher, tuple(value.shape))
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        _framed(hasher, "tensor-bytes", memoryview(raw))
        return
    if isinstance(value, Mapping):
        _framed(hasher, "mapping-size", str(len(value)).encode("ascii"))
        for key in sorted(value, key=lambda item: (type(item).__qualname__, repr(item))):
            _update_digest(hasher, key)
            _update_digest(hasher, value[key])
        return
    if isinstance(value, tuple):
        _framed(hasher, "tuple-size", str(len(value)).encode("ascii"))
        for item in value:
            _update_digest(hasher, item)
        return
    if isinstance(value, list):
        _framed(hasher, "list-size", str(len(value)).encode("ascii"))
        for item in value:
            _update_digest(hasher, item)
        return
    if isinstance(value, (set, frozenset)):
        _framed(hasher, "set-size", str(len(value)).encode("ascii"))
        for item in sorted(value, key=lambda item: (type(item).__qualname__, repr(item))):
            _update_digest(hasher, item)
        return
    raise TypeError(f"unsupported gate digest value: {type(value).__name__}")


def _digest(value: object) -> str:
    hasher = hashlib.sha256()
    _update_digest(hasher, value)
    return hasher.hexdigest()


def _logical_loader_state(value: object) -> object:
    state = copy.deepcopy(value)

    def normalize(node: object) -> None:
        if isinstance(node, dict):
            sampler = node.get("_index_sampler_state")
            if isinstance(sampler, dict) and isinstance(sampler.get("position"), int):
                for field in _LOADER_DIAGNOSTIC_FIELDS:
                    if field in node:
                        node[field] = sampler["position"]
            for child in node.values():
                normalize(child)
        elif isinstance(node, (tuple, list)):
            for child in node:
                normalize(child)

    normalize(state)
    return state


def _run_state(run: object, *, logical_loader: bool) -> dict[str, str]:
    checkpoint_state = run.checkpoint_state
    loader_state = checkpoint_state.dataloader.state_dict()
    if logical_loader:
        loader_state = _logical_loader_state(loader_state)
    optional = {
        name: None if component is None else component.state_dict()
        for name, component in checkpoint_state._optional_stateful.items()
    }
    return {
        "student": _digest(run.roles.student.module.state_dict()),
        "fake_score": _digest(run.roles.fake_score.module.state_dict()),
        "student_optimizer": _digest(checkpoint_state.optimizers[0].state_dict()),
        "fake_score_optimizer": _digest(checkpoint_state.optimizers[1].state_dict()),
        "engine": _digest(checkpoint_state.engine.state_dict()),
        "progress": _digest(checkpoint_state.progress.state_dict()),
        "dataloader": _digest(loader_state),
        "objective_generator": _digest(
            checkpoint_state.objective_generator.get_state()
        ),
        "optional_state": _digest(optional),
        "torch_cpu_rng": _digest(torch.get_rng_state()),
        "torch_cuda_rng": _digest(torch.cuda.get_rng_state_all()),
        "python_rng": _digest(random.getstate()),
        "resume_identity": checkpoint_state.identity_digest,
    }


def _assert_same(
    expected: Mapping[str, str],
    actual: Mapping[str, str],
    *,
    label: str,
) -> int:
    if set(expected) != set(actual):
        raise AssertionError(f"{label} state component inventory differs")
    mismatches = [name for name in expected if expected[name] != actual[name]]
    if mismatches:
        raise AssertionError(f"{label} state digests differ: {mismatches}")
    return len(expected)


def _summary_endpoint(summary: object) -> dict[str, object]:
    return {
        "final_step": summary.final_step,
        "student_optimizer_steps": summary.student_optimizer_steps,
        "fake_score_optimizer_steps": summary.fake_score_optimizer_steps,
        "final_generator_loss": summary.final_generator_loss,
        "final_fake_score_loss": summary.final_fake_score_loss,
    }


def main() -> int:
    args = _arguments()
    if isinstance(args.steps, bool) or args.steps <= 0:
        raise ValueError("--steps must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real SANA SiD gate requires a CUDA device")
    recipe_path = args.recipe.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if work_dir.exists():
        raise FileExistsError(f"gate work directory already exists: {work_dir}")
    for path, kind in ((model_root, "directory"), (cache, "directory")):
        if not path.is_dir():
            raise FileNotFoundError(f"required {kind} does not exist: {path}")
    if not manifest.is_file():
        raise FileNotFoundError(f"required manifest does not exist: {manifest}")

    source_recipe = PostTrainingRecipe.from_file(recipe_path)
    recipe = _gate_recipe(
        source_recipe,
        manifest=manifest,
        cache=cache,
        work_dir=work_dir,
        steps=args.steps,
        height=args.height,
        width=args.width,
    )
    work_dir.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    role_paths = {
        "student": model_root,
        "teacher": model_root,
        "fake_score": model_root,
    }
    common = {
        "base_dir": work_dir,
        "device": device,
        "local_role_paths": role_paths,
        "verify_media_hashes": True,
        "audit_cache_on_open": True,
        "verify_cache_on_read": True,
        "fused_adamw": False,
        "initialization_seed": args.seed,
    }

    split = materialize_sana_sid_training_run(
        recipe,
        output_dir=work_dir / "split-run",
        **common,
    )
    try:
        student_before = _digest(split.roles.student.module.state_dict())
        fake_before = _digest(split.roles.fake_score.module.state_dict())
        split_summary = split.run(max_steps=args.steps)
        split_state = _run_state(split, logical_loader=False)
        if split_state["student"] == student_before:
            raise AssertionError("real SANA SiD did not update the student")
        if split_state["fake_score"] == fake_before:
            raise AssertionError("real SANA SiD did not update the fake-score model")
        checkpoint_path = split.checkpointer.root / f"step-{split_summary.final_step:08d}"
        checkpoint = split.checkpointer.inspect(checkpoint_path)
        artifact = split.export_student()
        if not isinstance(artifact, FullModelArtifact):
            raise TypeError("real SANA SiD gate requires a Safetensors full-model export")
        asset_digests = dict(split.roles.asset_digests)
        data_identity = dict(split.data_identity)
    finally:
        split.close()
    del split
    _release()

    resumed = materialize_sana_sid_training_run(
        recipe,
        output_dir=work_dir / "resumed-run",
        resume_checkpoint=checkpoint.path,
        **common,
    )
    try:
        if resumed.resume_artifact is None:
            raise AssertionError("real SANA SiD resume did not bind a checkpoint")
        if resumed.resume_artifact.manifest_sha256 != checkpoint.manifest_sha256:
            raise AssertionError("real SANA SiD resumed a different checkpoint")
        immediate_components = _assert_same(
            split_state,
            _run_state(resumed, logical_loader=False),
            label="immediate resume",
        )
        resumed_summary = resumed.run(max_steps=1)
        resumed_state = _run_state(resumed, logical_loader=True)
    finally:
        resumed.close()
    del resumed
    _release()

    continuous = materialize_sana_sid_training_run(
        recipe,
        output_dir=work_dir / "continuous-run",
        **common,
    )
    try:
        continuous_summary = continuous.run(max_steps=args.steps + 1)
        continuous_state = _run_state(continuous, logical_loader=True)
    finally:
        continuous.close()
    del continuous
    _release()

    continuation_components = _assert_same(
        continuous_state,
        resumed_state,
        label="continuous resume",
    )
    continuous_endpoint = _summary_endpoint(continuous_summary)
    resumed_endpoint = _summary_endpoint(resumed_summary)
    if continuous_endpoint != resumed_endpoint:
        raise AssertionError(
            f"continuous-resume summary differs: {continuous_endpoint} != {resumed_endpoint}"
        )
    if not all(
        math.isfinite(float(value))
        for value in (
            split_summary.final_generator_loss,
            split_summary.final_fake_score_loss,
            resumed_summary.final_generator_loss,
            resumed_summary.final_fake_score_loss,
        )
    ):
        raise FloatingPointError("real SANA SiD returned non-finite losses")

    preparation, restored = build_local_diffusers_sana_sid_adapter(
        str(model_root),
        device=device,
        dtype={
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[recipe.runtime.param_dtype],
        parameter_dtype=torch.float32,
        checkpoint_identity=recipe.model.checkpoint,
        load_conditioner=False,
    )
    load_full_model(restored.module, artifact.path)
    restored_digest = _digest(restored.module.state_dict())
    if restored_digest != split_state["student"]:
        raise AssertionError("exported SANA SiD student differs after fresh-model reload")
    del preparation, restored
    _release()

    report = {
        "schema": "worldfoundry-sana-sid-roundtrip-gate",
        "execution_owner": recipe.execution_owner,
        "recipe_digest": recipe.digest,
        "model": {
            "recipe": recipe.model.recipe,
            "root": str(model_root),
            "role_asset_sha256": asset_digests,
            "trainable_parameter_dtype": "float32",
            "compute_dtype": recipe.runtime.param_dtype,
        },
        "algorithm": recipe.to_dict()["algorithm"],
        "data": data_identity,
        "summary": _summary_endpoint(split_summary),
        "updates": {
            "student_changed": True,
            "fake_score_changed": True,
        },
        "checkpoint": {
            "path": str(checkpoint.path),
            "manifest_sha256": checkpoint.manifest_sha256,
            "identity_digest": checkpoint.identity_digest,
            "immediate_resume": {
                "exact": True,
                "state_components": immediate_components,
            },
            "continuous_resume": {
                "exact": True,
                "state_components": continuation_components,
                "summary": resumed_endpoint,
                "normalized_loader_diagnostics": list(_LOADER_DIAGNOSTIC_FIELDS),
            },
        },
        "artifact": {
            "path": str(artifact.path),
            "manifest_sha256": artifact.manifest_sha256,
            "file_sha256": dict(artifact.file_digests),
            "fresh_model_reload_exact": True,
        },
        "system": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    replace_json_atomic(work_dir / "gate_result.json", report, root=work_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
