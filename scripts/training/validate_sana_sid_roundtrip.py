#!/usr/bin/env python3
"""Run a real-weight SANA SiD update, exact resume, and export-reload gate."""

from __future__ import annotations

import argparse
import copy
import gc
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
from worldfoundry.training.state_comparison import (
    assert_state_equal,
    file_size_inventory,
    snapshot_state,
    state_changed,
)
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


def _run_state(run: object, *, logical_loader: bool) -> dict[str, object]:
    checkpoint_state = run.checkpoint_state
    loader_state = checkpoint_state.dataloader.state_dict()
    if logical_loader:
        loader_state = _logical_loader_state(loader_state)
    optional = {
        name: None if component is None else component.state_dict()
        for name, component in checkpoint_state._optional_stateful.items()
    }
    return {
        "student": snapshot_state(run.roles.student.module.state_dict()),
        "fake_score": snapshot_state(run.roles.fake_score.module.state_dict()),
        "student_optimizer": snapshot_state(checkpoint_state.optimizers[0].state_dict()),
        "fake_score_optimizer": snapshot_state(checkpoint_state.optimizers[1].state_dict()),
        "engine": snapshot_state(checkpoint_state.engine.state_dict()),
        "progress": snapshot_state(checkpoint_state.progress.state_dict()),
        "dataloader": snapshot_state(loader_state),
        "objective_generator": snapshot_state(checkpoint_state.objective_generator.get_state()),
        "optional_state": snapshot_state(optional),
        "torch_cpu_rng": snapshot_state(torch.get_rng_state()),
        "torch_cuda_rng": snapshot_state(torch.cuda.get_rng_state_all()),
        "python_rng": snapshot_state(random.getstate()),
    }


def _assert_same(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    label: str,
) -> int:
    if set(expected) != set(actual):
        raise AssertionError(f"{label} state component inventory differs")
    for name in expected:
        assert_state_equal(expected[name], actual[name], path=f"{label}.{name}")
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
        "audit_cache_on_open": True,
        "fused_adamw": False,
        "initialization_seed": args.seed,
    }

    split = materialize_sana_sid_training_run(
        recipe,
        output_dir=work_dir / "split-run",
        **common,
    )
    try:
        student_before = snapshot_state(split.roles.student.module.state_dict())
        fake_before = snapshot_state(split.roles.fake_score.module.state_dict())
        split_summary = split.run(max_steps=args.steps)
        split_state = _run_state(split, logical_loader=False)
        if not state_changed(student_before, split_state["student"]):
            raise AssertionError("real SANA SiD did not update the student")
        if not state_changed(fake_before, split_state["fake_score"]):
            raise AssertionError("real SANA SiD did not update the fake-score model")
        checkpoint_path = split.checkpointer.root / f"step-{split_summary.final_step:08d}"
        checkpoint = split.checkpointer.inspect(checkpoint_path)
        artifact = split.export_student()
        if not isinstance(artifact, FullModelArtifact):
            raise TypeError("real SANA SiD gate requires a Safetensors full-model export")
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
        if resumed.resume_artifact.path.resolve() != checkpoint.path.resolve():
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
    assert_state_equal(
        split_state["student"],
        restored.module.state_dict(),
        path="exported student",
    )
    del preparation, restored
    _release()

    report = {
        "schema": "worldfoundry-sana-sid-roundtrip-gate",
        "execution_owner": recipe.execution_owner,
        "recipe": recipe.to_dict(),
        "model": {
            "recipe": recipe.model.recipe,
            "root": str(model_root),
            "role_paths": {name: str(path) for name, path in role_paths.items()},
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
            "files": file_size_inventory(checkpoint.path),
            "identity": dict(checkpoint.identity),
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
            "files": file_size_inventory(artifact.path),
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
