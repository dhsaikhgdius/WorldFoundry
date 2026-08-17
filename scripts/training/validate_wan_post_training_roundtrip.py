#!/usr/bin/env python3
"""Run a real-weight Wan DMD update, exact resume, and export-reload check."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
from worldfoundry.base_models.diffusion_model.recipes.registry import (
    default_native_diffusion_registry,
)
from worldfoundry.training.data import (
    TRAINING_SAMPLE_SCHEMA,
    materialize_wan_training_cache,
)
from worldfoundry.training.engine import materialize_wan_dmd_training_run
from worldfoundry.training.engine.wan.roles import load_wan_role_adapter
from worldfoundry.training.recipes import PostTrainingRecipe
from worldfoundry.training.safety import PromptSafetyAudit
from worldfoundry.training.safety.shieldgemma import SHIELDGEMMA_PROMPT_POLICIES
from worldfoundry.training.tuning import PeftAdapterArtifact, load_peft_adapter


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fixture_audit(prompt: str) -> PromptSafetyAudit:
    return PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={name: 0.0 for name in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )


def _prepare_fixture(source_root: Path, work_dir: Path) -> tuple[Path, PromptSafetyAudit]:
    source_manifest = source_root / "manifest.jsonl"
    try:
        row = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Wan gate fixture manifest: {source_manifest}") from error
    if not isinstance(row, dict) or row.get("schema") != TRAINING_SAMPLE_SCHEMA:
        raise ValueError("Wan gate fixture must contain one native training sample")
    prompt = str(row["prompt"])
    audit = _fixture_audit(prompt)
    media = row.get("media")
    if not isinstance(media, dict):
        raise ValueError("Wan gate fixture media descriptor is invalid")
    source_video = source_root / str(media["uri"])
    video_path = work_dir / source_video.name
    shutil.copy2(source_video, video_path)
    kept = {
        key: row[key]
        for key in (
            "schema",
            "sample_id",
            "task",
            "prompt",
            "width",
            "height",
            "num_frames",
            "fps",
            "conditions",
            "split",
        )
    }
    kept["media"] = {
        "uri": video_path.name,
        "size_bytes": video_path.stat().st_size,
        "mime_type": media.get("mime_type", "video/x-matroska"),
    }
    kept["safety"] = {
        "prompt_safe": audit.safe,
        "model_revision": audit.model_revision,
    }
    manifest_path = work_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(kept, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, audit


def _recipe(
    *,
    work_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "schema": "worldfoundry-post-training",
            "execution_owner": "worldfoundry-native",
            "run": {
                "id": "wan-1p3b-real-dmd-gate",
                "output_dir": str(work_dir / "run"),
            },
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "default",
                "options": {"vae_tiled": False},
            },
            "tuning": {
                "mode": "lora",
                "preset": "wan-attention",
                "rank": 4,
                "alpha": 4,
                "dropout": 0.0,
            },
            "data": {
                "manifest": str(manifest_path),
                "cache": str(cache_dir),
                "max_latent_tokens_per_microbatch": 128,
                "split": "train",
                "shuffle": False,
                "shuffle_seed": 42,
                "tail_policy": "pad",
                "options": {
                    "video_buckets": [
                        {
                            "num_frames": 5,
                            "height": 64,
                            "width": 64,
                            "conditioning_layout": "umt5-sequence",
                            "tasks": ["t2v"],
                        }
                    ],
                    "bucket_policy": {
                        "allow_spatial_upscale": False,
                        "allow_temporal_padding": False,
                    },
                    "decode": {
                        "frame_sampling": "uniform-full",
                        "interpolation": "bicubic",
                        "value_range": "minus-one-one",
                        "decoder_thread_type": "auto",
                        "verify_manifest_frame_count": True,
                        "verify_manifest_geometry": True,
                        "fps_tolerance": 0.01,
                    },
                    "num_workers": 0,
                    "pin_memory": True,
                    "persistent_workers": False,
                    "snapshot_every_n_steps": 1,
                },
            },
            "algorithm": {
                "type": "dmd",
                "student_timesteps": [1000, 757, 522],
                "student_sigmas": [1.0, 0.757, 0.522],
                "real_score_checkpoint": ("Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a"),
                "fake_score_checkpoint": ("Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a"),
                "num_train_timesteps": 1000,
                "score_min_sigma": 0.02,
                "score_max_sigma": 0.98,
                "score_flow_shift": 8.0,
                "teacher_guidance_scale": 3.5,
                "generator_update_interval": 5,
                "student_scheduler_cadence": "iteration",
                "normalization_epsilon": 0.0,
                "shared_score_timestep": True,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 2.0e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 8,
            },
            "fake_score_optimizer": {
                "type": "adamw",
                "learning_rate": 2.0e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 8,
            },
            "runtime": {
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
                "activation_checkpoint": "full",
                "compile": False,
            },
            "distributed": {
                "backend": "single",
                "dp_replicate": 1,
                "dp_shard": "auto",
                "cp": 1,
                "tp": 1,
            },
            "checkpoint": {
                "save_every_steps": 1,
                "async": False,
                "export_every_steps": 0,
            },
            "export": {"format": "peft"},
        }
    )


def _trainable_state(application) -> dict[str, torch.Tensor]:
    names = set(application.trainable_parameter_names)
    state = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in application.model.named_parameters()
        if name in names
    }
    if set(state) != names:
        raise RuntimeError("PEFT trainable parameter inventory differs from the live model")
    return state


def _delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, object]:
    if set(before) != set(after):
        raise RuntimeError("trainable parameter names changed during DMD")
    differences = {name: after[name] - before[name] for name in before}
    changed = sum(int(bool(torch.count_nonzero(value))) for value in differences.values())
    l2 = math.sqrt(sum(float(value.double().square().sum().item()) for value in differences.values()))
    maximum = max(float(value.abs().max().item()) for value in differences.values())
    return {
        "changed_parameter_tensors": changed,
        "parameter_delta_l2": l2,
        "parameter_delta_max_abs": maximum,
    }


def _snapshot(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return copy.deepcopy(value)


def _assert_exact(expected: object, actual: object, *, path: str = "state") -> int:
    """Assert exact nested parity and return the compared tensor count."""

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path} changed from tensor to {type(actual).__name__}")
        if expected.dtype != actual.dtype or tuple(expected.shape) != tuple(actual.shape):
            raise AssertionError(
                f"{path} tensor contract changed: "
                f"{expected.dtype}/{tuple(expected.shape)} != "
                f"{actual.dtype}/{tuple(actual.shape)}"
            )
        if not torch.equal(expected.cpu(), actual.detach().cpu()):
            raise AssertionError(f"{path} tensor bytes differ")
        return 1
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            actual_keys = set(actual) if isinstance(actual, Mapping) else type(actual).__name__
            raise AssertionError(f"{path} mapping keys differ: {set(expected)} != {actual_keys}")
        return sum(_assert_exact(expected[key], actual[key], path=f"{path}.{key}") for key in expected)
    if isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f"{path} sequence contract differs")
        return sum(
            _assert_exact(left, right, path=f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        )
    if type(expected) is not type(actual) or expected != actual:
        raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")
    return 0


def _logical_checkpoint_state(value: object) -> object:
    """Canonicalize only TorchData's iterator-local diagnostic counters.

    A stateful custom batch sampler owns the actual next-sample position.  At
    an epoch boundary, TorchData can represent the same restored sampler with
    either a continued iterator counter or a freshly-created iterator counter.
    Normalize those two counters to the sampler position for parity auditing;
    every sampler queue, worker snapshot, RNG state, optimizer tensor, and
    model tensor remains part of the exact comparison.
    """

    snapshot = _snapshot(value)
    if not isinstance(snapshot, dict):
        raise TypeError("checkpoint snapshot must be a mapping")
    runtime_by_rank = snapshot.get("runtime_by_rank")
    if not isinstance(runtime_by_rank, dict):
        raise TypeError("checkpoint snapshot is missing runtime_by_rank")
    for runtime in runtime_by_rank.values():
        if not isinstance(runtime, dict) or not isinstance(runtime.get("dataloader"), str):
            raise TypeError("rank runtime must contain serialized dataloader state")
        loader = json.loads(runtime["dataloader"])
        source = loader.get("source") if isinstance(loader, dict) else None
        sampler = source.get("_index_sampler_state") if isinstance(source, dict) else None
        if isinstance(sampler, dict) and isinstance(sampler.get("position"), int):
            position = sampler["position"]
            for field in ("_num_yielded", "_sampler_iter_yielded"):
                if field not in source:
                    raise TypeError(f"TorchData state is missing {field}")
                source[field] = position
        runtime["dataloader"] = json.dumps(
            loader,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return snapshot


def _summary_dict(summary: object) -> dict[str, object]:
    fields = getattr(summary, "__dataclass_fields__", None)
    if not isinstance(fields, dict):
        raise TypeError("DMD summary must be a dataclass instance")
    return {name: getattr(summary, name) for name in fields}


def _summary_endpoint(summary: object) -> dict[str, object]:
    values = _summary_dict(summary)
    return {
        name: values[name]
        for name in (
            "final_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "final_generator_loss",
            "final_fake_score_loss",
        )
    }


def _reload_exported_student(
    *,
    native_recipe: object,
    checkpoint: object,
    artifact: PeftAdapterArtifact,
    expected_names: set[str],
    num_train_timesteps: int,
) -> dict[str, torch.Tensor]:
    adapter = load_wan_role_adapter(
        assembler=NativeDiffusionAssembler(),
        native_recipe=native_recipe,
        checkpoint=checkpoint,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        num_train_timesteps=num_train_timesteps,
        gradient_checkpointing=True,
        force_torch_attention=True,
    )
    restored = load_peft_adapter(adapter.trainable_module, artifact.path)
    state = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in restored.named_parameters()
        if name in expected_names
    }
    if set(state) != expected_names:
        raise AssertionError(
            "PEFT reload parameter inventory differs from the trained student: "
            f"missing={sorted(expected_names - set(state))}, "
            f"unexpected={sorted(set(state) - expected_names)}"
        )
    return state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.steps < 5:
        raise ValueError("--steps must be at least five to exercise the student update cadence")
    if not torch.cuda.is_available():
        raise RuntimeError("the real Wan DMD gate requires CUDA")
    model_root = args.model_root.expanduser().resolve()
    fixture_root = args.fixture_root.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if work_dir.exists():
        raise FileExistsError(f"Wan DMD gate output already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    started = time.perf_counter()
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(103)
    torch.cuda.reset_peak_memory_stats()

    manifest_path, audit = _prepare_fixture(fixture_root, work_dir)
    cache_dir = work_dir / "cache"
    recipe = _recipe(
        work_dir=work_dir,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
    )
    component_overrides = {name: str(model_root) for name in ("dit", "text-encoder", "tokenizer", "vae")}
    cache = materialize_wan_training_cache(
        recipe,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        device="cuda",
        checkpoint_overrides=component_overrides,
        safety_audits=(audit,),
        verify_media_files=True,
    )
    _release()

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    resolved = NativeDiffusionAssembler.resolve_checkpoints(
        native_recipe,
        {"dit": str(model_root)},
    )
    role_overrides = {name: resolved["dit"] for name in ("student", "real-score", "fake-score")}
    run = materialize_wan_dmd_training_run(
        recipe,
        device="cuda",
        output_dir=work_dir / "run",
        audited_role_overrides=role_overrides,
        verify_media_files=True,
        audit_cache_on_open=True,
        verify_cache_on_read=True,
        force_torch_attention=True,
        fused_adamw=False,
        initialization_seed=107,
    )
    try:
        assert run.roles.student_peft is not None
        assert run.roles.fake_score_peft is not None
        student_before = _trainable_state(run.roles.student_peft)
        fake_before = _trainable_state(run.roles.fake_score_peft)
        checkpoint_summary = run.run(max_steps=args.steps)
        student_at_checkpoint = _trainable_state(run.roles.student_peft)
        fake_at_checkpoint = _trainable_state(run.roles.fake_score_peft)
        student_delta = _delta(student_before, student_at_checkpoint)
        fake_delta = _delta(fake_before, fake_at_checkpoint)
        artifact = run.export_student()
        if not isinstance(artifact, PeftAdapterArtifact):
            raise TypeError("real Wan DMD gate expected a PEFT student artifact")
        losses = (
            checkpoint_summary.final_generator_loss,
            checkpoint_summary.final_fake_score_loss,
        )
        if not all(math.isfinite(value) for value in losses):
            raise FloatingPointError(f"real Wan DMD returned non-finite losses: {losses}")
        if student_delta["changed_parameter_tensors"] <= 0:
            raise AssertionError("real Wan DMD did not update any student tensors")
        if fake_delta["changed_parameter_tensors"] <= 0:
            raise AssertionError("real Wan DMD did not update any fake-score tensors")
        checkpoint_path = run.checkpointer.root / f"step-{checkpoint_summary.final_step:08d}"
        checkpoint = run.checkpointer.inspect(checkpoint_path)
        student_role_checkpoint = run.roles.student_checkpoint
        checkpoint_state = _snapshot(run.checkpoint_state.state_dict())

        # Fork the continuation from the exact live state that produced the
        # checkpoint.  Re-training an independent prefix is not a resume
        # oracle: ordinary CUDA kernels can differ by a few low-order bits
        # between separately materialized runs before checkpointing occurs.
        continuous_summary = run.run(max_steps=1)
        continuous_student = _trainable_state(run.roles.student_peft)
        continuous_fake = _trainable_state(run.roles.fake_score_peft)
        continuous_state = _snapshot(run.checkpoint_state.state_dict())

    finally:
        run.close()
    del run
    _release()

    resumed = materialize_wan_dmd_training_run(
        recipe,
        device="cuda",
        output_dir=work_dir / "resumed-run",
        resume_checkpoint=checkpoint.path,
        audited_role_overrides=role_overrides,
        verify_media_files=True,
        audit_cache_on_open=True,
        verify_cache_on_read=True,
        force_torch_attention=True,
        fused_adamw=False,
        initialization_seed=107,
    )
    try:
        if resumed.resume_artifact is None:
            raise AssertionError("real Wan DMD resume did not bind a checkpoint artifact")
        if resumed.resume_artifact != checkpoint:
            raise AssertionError("real Wan DMD resumed a different checkpoint")
        assert resumed.roles.student_peft is not None
        assert resumed.roles.fake_score_peft is not None
        immediate_student_tensors = _assert_exact(
            student_at_checkpoint,
            _trainable_state(resumed.roles.student_peft),
            path="resume.student",
        )
        immediate_fake_tensors = _assert_exact(
            fake_at_checkpoint,
            _trainable_state(resumed.roles.fake_score_peft),
            path="resume.fake_score",
        )
        immediate_state_tensors = _assert_exact(
            checkpoint_state,
            _snapshot(resumed.checkpoint_state.state_dict()),
            path="resume.checkpoint_state",
        )

        resumed_summary = resumed.run(max_steps=1)
        resumed_student_tensors = _assert_exact(
            continuous_student,
            _trainable_state(resumed.roles.student_peft),
            path="continuation.student",
        )
        resumed_fake_tensors = _assert_exact(
            continuous_fake,
            _trainable_state(resumed.roles.fake_score_peft),
            path="continuation.fake_score",
        )
        resumed_state_tensors = _assert_exact(
            _logical_checkpoint_state(continuous_state),
            _logical_checkpoint_state(resumed.checkpoint_state.state_dict()),
            path="continuation.checkpoint_state",
        )
        _assert_exact(
            _summary_endpoint(continuous_summary),
            _summary_endpoint(resumed_summary),
            path="continuation.endpoint",
        )
    finally:
        resumed.close()
    del resumed
    _release()

    restored_student = _reload_exported_student(
        native_recipe=native_recipe,
        checkpoint=student_role_checkpoint,
        artifact=artifact,
        expected_names=set(student_at_checkpoint),
        num_train_timesteps=recipe.algorithm.num_train_timesteps,
    )
    artifact_tensors = _assert_exact(
        student_at_checkpoint,
        restored_student,
        path="artifact.student",
    )
    del restored_student
    _release()

    report = {
        "schema": "worldfoundry-wan-dmd-roundtrip-gate",
        "execution_owner": recipe.execution_owner,
        "model": {
            "recipe": native_recipe.model_id,
            "asset_revision": native_recipe.checkpoints["dit"].revision,
            "root": str(model_root),
        },
        "algorithm": recipe.to_dict()["algorithm"],
        "cache": {
            "index": cache.index.to_dict(),
            "entry": cache.entries[0].to_dict(),
            "unconditional_conditioning": cache.unconditional_conditioning.to_dict(),
        },
        "summary": _summary_dict(checkpoint_summary),
        "student_update": student_delta,
        "fake_score_update": fake_delta,
        "checkpoint": {
            "path": str(checkpoint.path),
            "identity": dict(checkpoint.identity),
            "file_size_bytes": dict(checkpoint.file_size_bytes),
            "immediate_resume": {
                "exact": True,
                "student_tensors": immediate_student_tensors,
                "fake_score_tensors": immediate_fake_tensors,
                "state_tensors": immediate_state_tensors,
            },
            "continuous_resume": {
                "exact": True,
                "oracle": "same-live-run-from-saved-boundary",
                "summary": _summary_dict(resumed_summary),
                "student_tensors": resumed_student_tensors,
                "fake_score_tensors": resumed_fake_tensors,
                "state_tensors": resumed_state_tensors,
                "logical_loader_position_owner": "native-token-batch-sampler",
                "normalized_torchdata_diagnostics": [
                    "_num_yielded",
                    "_sampler_iter_yielded",
                ],
            },
        },
        "artifact": {
            "path": str(artifact.path),
            "file_size_bytes": dict(artifact.file_size_bytes),
            "reload": {
                "exact": True,
                "student_tensors": artifact_tensors,
            },
        },
        "system": {
            "device_name": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    report_path = work_dir / "gate_result.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
