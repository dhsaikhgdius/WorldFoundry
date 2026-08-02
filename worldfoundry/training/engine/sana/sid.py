"""Materialize local SANA Score Identity Distillation on native infrastructure."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.core.io.file_utils import file_sha256
from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.rollout_manifest import RolloutPromptDataset
from worldfoundry.training.data.sampler import DeterministicDistributedSampler
from worldfoundry.training.data.sana_cache import SanaCachedDataset, collate_sana_cached_samples
from worldfoundry.training.data.shared_conditioning import SharedConditioningStore
from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.distributed.parallel import DistributedTrainingContext, ParallelPlan
from worldfoundry.training.models.sana_sid import build_local_diffusers_sana_sid_adapter
from worldfoundry.training.post_training.distillation.sid.builder import build_native_sid_training_stack
from worldfoundry.training.post_training.distillation.sid.session import NativeSIDTrainingSession
from worldfoundry.training.post_training.shared.distributed import PostTrainingParallelContext
from worldfoundry.training.recipes.post_training.algorithms.sid import SIDAlgorithmSpec
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ..artifacts import create_run_directory
from .cache import audit_sana_cache_against_manifest, validate_sana_cache_contract
from .scm_ladd_data import audit_sana_scm_ladd_unconditional
from .sid_data import SanaSIDDataLoader, collate_sana_sid_prompts
from .sid_roles import SanaSIDRoleBundle, SanaSIDTrainableRoles
from .sid_run import SANA_SID_RUN_SCHEMA, SanaSIDTrainingRun

_DATA_OPTIONS = frozenset(
    {
        "height",
        "width",
        "microbatch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "snapshot_every_n_steps",
    }
)


def _dtype(value: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _strict_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _seed(seed: int) -> None:
    resolved = int(seed) % (2**63 - 1)
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def _validate_recipe(recipe: PostTrainingRecipe) -> SIDAlgorithmSpec:
    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, SIDAlgorithmSpec):
        raise TypeError("SANA SiD requires algorithm.type='sid'")
    if not recipe.model.recipe.startswith("sana-"):
        raise ValueError("SANA SiD requires a SANA image model recipe")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("SANA SiD supports single or FSDP2 execution")
    if recipe.distributed.backend == "fsdp2" and recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("multi-rank SANA SiD requires data.tail_policy='drop' or 'pad'")
    if recipe.runtime.activation_checkpoint != "none":
        raise ValueError("local SANA SiD activation checkpointing is not implemented")
    if recipe.runtime.compile:
        raise ValueError("local SANA SiD does not support torch.compile")
    if recipe.runtime.reduce_dtype != "float32":
        raise ValueError("SANA SiD objective reduction must use float32")
    if recipe.optimizer.type != "adamw" or (
        recipe.fake_score_optimizer is None or recipe.fake_score_optimizer.type != "adamw"
    ):
        raise ValueError("SANA SiD requires AdamW for student and fake-score")
    if recipe.tuning.mode != "full":
        raise ValueError("local Diffusers SANA SiD currently requires full student tuning")
    if recipe.algorithm.diffusion_gan_enabled:
        raise ValueError(
            "local Diffusers SANA has no discriminator feature head; inject a discriminator-capable "
            "fake-score adapter through the model-neutral SiD builder"
        )
    return recipe.algorithm


def _role_paths(
    recipe: PostTrainingRecipe,
    base_dir: Path,
    overrides: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    raw = dict(overrides or {})
    unknown = sorted(set(raw) - {"student", "teacher", "fake_score"})
    if unknown:
        raise ValueError(f"unknown SANA SiD role paths: {unknown}")
    identities = {
        "student": recipe.model.checkpoint,
        "teacher": recipe.algorithm.teacher_checkpoint,
        "fake_score": recipe.algorithm.fake_score_checkpoint,
    }
    result: dict[str, Path] = {}
    for role, identity in identities.items():
        path = Path(raw.get(role, identity)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"SANA SiD {role} local model directory does not exist: {path}")
        result[role] = path
    return result


def _asset_digest(
    path: Path,
    *,
    conditioner: bool,
    hash_cache: dict[Path, str],
) -> str:
    def model_files(directory: Path, *, stem: str) -> list[Path]:
        single = directory / f"{stem}.safetensors"
        index = directory / f"{stem}.safetensors.index.json"
        if single.exists() and index.exists():
            raise ValueError(f"ambiguous local SANA model assets in {directory}")
        if single.is_file():
            return [single]
        if not index.is_file():
            raise FileNotFoundError(
                f"local SANA model assets lack {single.name} or {index.name}: {directory}"
            )
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid local SANA Safetensors index: {index}") from error
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"local SANA Safetensors index has no weight_map: {index}")
        raw_shard_names = list(weight_map.values())
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in raw_shard_names
        ):
            raise ValueError(f"local SANA Safetensors index has unsafe shard names: {index}")
        shard_names = sorted(set(raw_shard_names))
        shards = [directory / name for name in shard_names]
        missing = [str(value) for value in shards if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"local SANA Safetensors index has missing shards: {missing}")
        return [index, *shards]

    transformer = path / "transformer"
    files = [transformer / "config.json"]
    files.extend(model_files(transformer, stem="diffusion_pytorch_model"))
    if conditioner:
        files.append(path / "model_index.json")
        text_encoder = path / "text_encoder"
        files.append(text_encoder / "config.json")
        files.extend(model_files(text_encoder, stem="model"))
        tokenizer = path / "tokenizer"
        if not tokenizer.is_dir():
            raise FileNotFoundError(f"local SANA tokenizer directory is missing: {tokenizer}")
        tokenizer_assets = tuple(tokenizer.rglob("*"))
        tokenizer_files = sorted(value for value in tokenizer_assets if value.is_file())
        if not tokenizer_files:
            raise FileNotFoundError(f"local SANA tokenizer directory is empty: {tokenizer}")
        files.extend(tokenizer_files)
    if any(not value.is_file() for value in files):
        missing = [str(value) for value in files if not value.is_file()]
        raise FileNotFoundError(f"local SANA asset set is incomplete: {missing}")

    def digest_file(value: Path) -> str:
        digest = hash_cache.get(value)
        if digest is None:
            digest = file_sha256(value)
            hash_cache[value] = digest
        return digest

    return canonical_sha256(
        {
            str(value.relative_to(path)): digest_file(value)
            for value in files
        }
    )


def _audit_prompt_geometry(
    prompts: RolloutPromptDataset,
    *,
    height: int,
    width: int,
) -> None:
    for record in prompts:
        generation = dict(record.generation)
        unknown = sorted(set(generation) - {"height", "width", "num_frames"})
        if unknown:
            raise ValueError(f"SANA SiD prompt {record.prompt_id!r} has unsupported generation fields: {unknown}")
        if int(generation.get("height", height)) != height or int(generation.get("width", width)) != width:
            raise ValueError(f"SANA SiD prompt {record.prompt_id!r} overrides the fixed run geometry")
        if int(generation.get("num_frames", 1)) != 1:
            raise ValueError("SANA SiD prompt manifests must request one image frame")


def materialize_sana_sid_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    local_role_paths: Mapping[str, str | Path] | None = None,
    verify_media_hashes: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SanaSIDTrainingRun:
    """Build independent local SANA roles, scalable execution, and exact resume."""

    _validate_recipe(recipe)
    root = Path(base_dir).expanduser().resolve()
    manifest_path = Path(recipe.data.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path.resolve()
    destination = Path(output_dir or recipe.run.output_dir).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    paths = _role_paths(recipe, root, local_role_paths)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("SANA SiD FSDP2 materialization requires CUDA")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device

    try:
        world_size = 1 if distributed_context is None else distributed_context.world_size
        rank = 0 if distributed_context is None else distributed_context.rank
        plan = ParallelPlan.resolve(recipe.distributed, world_size=world_size)
        create_run_directory(destination, distributed_context)
        options = dict(recipe.data.options)
        unknown_options = sorted(set(options) - _DATA_OPTIONS)
        if unknown_options:
            raise ValueError(f"unsupported SANA SiD data options: {unknown_options}")
        height = _positive_int(options.pop("height", None), field_name="data.options.height")
        width = _positive_int(options.pop("width", None), field_name="data.options.width")
        microbatch_size = _positive_int(
            options.pop("microbatch_size", 1),
            field_name="data.options.microbatch_size",
        )
        workers = int(options.pop("num_workers", 0))
        pin_memory = _strict_bool(
            options.pop("pin_memory", resolved_device.type == "cuda"),
            field_name="data.options.pin_memory",
        )
        persistent_workers = _strict_bool(
            options.pop("persistent_workers", False),
            field_name="data.options.persistent_workers",
        )
        prefetch = options.pop("prefetch_factor", None)
        snapshot = _positive_int(
            options.pop("snapshot_every_n_steps", 1),
            field_name="data.options.snapshot_every_n_steps",
        )
        assert not options

        dtype = _dtype(recipe.runtime.param_dtype)
        base_seed = int(recipe.data.shuffle_seed if initialization_seed is None else initialization_seed)
        _seed(base_seed)
        prompt_only = recipe.data.cache is None
        # AdamW owns the student and fake-score parameters directly.  Keep
        # FP32 masters and let the prediction adapter/FSDP policy cast forward
        # compute; low learning rates can otherwise quantize to zero in BF16.
        student_preparation, student = build_local_diffusers_sana_sid_adapter(
            str(paths["student"]),
            device=resolved_device,
            dtype=dtype,
            parameter_dtype=torch.float32,
            checkpoint_identity=recipe.model.checkpoint,
            load_conditioner=prompt_only,
        )
        student.module.requires_grad_(True)
        teacher_preparation, teacher = build_local_diffusers_sana_sid_adapter(
            str(paths["teacher"]),
            device=resolved_device,
            dtype=dtype,
            checkpoint_identity=recipe.algorithm.teacher_checkpoint,
            load_conditioner=False,
        )
        teacher.module.requires_grad_(False)
        teacher.module.eval()
        fake_preparation, fake_score = build_local_diffusers_sana_sid_adapter(
            str(paths["fake_score"]),
            device=resolved_device,
            dtype=dtype,
            parameter_dtype=torch.float32,
            checkpoint_identity=recipe.algorithm.fake_score_checkpoint,
            load_conditioner=False,
        )
        fake_score.module.requires_grad_(True)

        student_fsdp: FSDP2Application | None = None
        teacher_fsdp: FSDP2Application | None = None
        fake_fsdp: FSDP2Application | None = None
        if distributed_context is not None:
            mesh = plan.build_device_mesh(resolved_device.type)
            reduce_dtype = _dtype(recipe.runtime.reduce_dtype)
            student_fsdp = apply_fsdp2(
                student_preparation,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
            )
            teacher_fsdp = apply_fsdp2_frozen_reference(
                teacher_preparation,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
            )
            fake_fsdp = apply_fsdp2(
                fake_preparation,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
            )

        if prompt_only:
            prompts = RolloutPromptDataset.from_file(manifest_path, split=recipe.data.split)
            _audit_prompt_geometry(prompts, height=height, width=width)
            sampler = DeterministicDistributedSampler(
                prompts,
                dataset_digest=prompts.dataset_digest,
                seed=recipe.data.shuffle_seed,
                shuffle=recipe.data.shuffle,
                rank=rank,
                world_size=world_size,
                tail_policy=recipe.data.tail_policy,
            )
            if len(sampler) < microbatch_size:
                raise ValueError("SANA SiD microbatch_size would leave this rank without a batch")
            source_loader = build_stateful_dataloader(
                prompts,
                sampler,
                batch_size=microbatch_size,
                collate_fn=collate_sana_sid_prompts,
                num_workers=workers,
                worker_seed=recipe.data.shuffle_seed,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=None if prefetch is None else int(prefetch),
                snapshot_every_n_steps=snapshot,
            )
            sid_loader = SanaSIDDataLoader(
                source_loader,
                adapter=student_preparation,
                height=height,
                width=width,
            )
            data_identity = {
                "kind": "prompt-only",
                "dataset_digest": prompts.dataset_digest,
                "height": height,
                "width": width,
            }
        else:
            cache_path = Path(recipe.data.cache or "").expanduser()
            if not cache_path.is_absolute():
                cache_path = root / cache_path
            cache_path = cache_path.resolve()
            manifest = TrainingManifestDataset.from_file(
                manifest_path,
                split=recipe.data.split,
                verify_files=True,
                verify_hashes=verify_media_hashes,
            )
            cache = SanaCachedDataset(
                cache_path,
                expected_dataset_digest=manifest.dataset_digest,
                audit_on_open=audit_cache_on_open,
                verify_on_read=verify_cache_on_read,
            )
            audit_sana_cache_against_manifest(cache, manifest)
            unconditional = SharedConditioningStore(cache_path).read("unconditional")
            audit_sana_scm_ladd_unconditional(unconditional, cache)
            contract_digest = validate_sana_cache_contract(
                recipe,
                student_preparation,
                cache,
                microbatch_size=microbatch_size,
            )
            sampler = DeterministicDistributedSampler(
                cache,
                dataset_digest=cache.dataset_digest,
                seed=recipe.data.shuffle_seed,
                shuffle=recipe.data.shuffle,
                rank=rank,
                world_size=world_size,
                tail_policy=recipe.data.tail_policy,
            )
            if len(sampler) < microbatch_size:
                raise ValueError("SANA SiD microbatch_size would leave this rank without a batch")
            source_loader = build_stateful_dataloader(
                cache,
                sampler,
                batch_size=microbatch_size,
                collate_fn=collate_sana_cached_samples,
                num_workers=workers,
                worker_seed=recipe.data.shuffle_seed,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=None if prefetch is None else int(prefetch),
                snapshot_every_n_steps=snapshot,
            )
            sid_loader = SanaSIDDataLoader(
                source_loader,
                adapter=student_preparation,
                height=height,
                width=width,
                unconditional=unconditional,
                include_real_latents=False,
            )
            data_identity = {
                "kind": "sana-cache",
                "dataset_digest": cache.dataset_digest,
                "cache_index_sha256": cache.index_sha256,
                "cache_contract_sha256": contract_digest,
                "unconditional_identity_sha256": unconditional.artifact.identity_sha256,
                "height": height,
                "width": width,
            }

        stack = build_native_sid_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            fake_score=fake_score,
            parallel_context=PostTrainingParallelContext.current(),
            fused_adamw=fused_adamw,
        )
        hash_cache: dict[Path, str] = {}
        asset_digests = {
            "student": _asset_digest(
                paths["student"],
                conditioner=prompt_only,
                hash_cache=hash_cache,
            ),
            "teacher": _asset_digest(
                paths["teacher"],
                conditioner=False,
                hash_cache=hash_cache,
            ),
            "fake_score": _asset_digest(
                paths["fake_score"],
                conditioner=False,
                hash_cache=hash_cache,
            ),
        }
        roles = SanaSIDRoleBundle(
            student_preparation=student_preparation,
            fake_score_preparation=fake_preparation,
            teacher_preparation=teacher_preparation,
            student=student,
            teacher=teacher,
            fake_score=fake_score,
            asset_digests=asset_digests,
            student_fsdp=student_fsdp,
            teacher_fsdp=teacher_fsdp,
            fake_score_fsdp=fake_fsdp,
        )
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        objective_generator = torch.Generator(device=resolved_device)
        objective_generator.manual_seed((base_seed + rank) % (2**63 - 1))
        checkpoint_model = SanaSIDTrainableRoles(student.module, fake_score.module)
        identity = {
            "schema": "worldfoundry-sana-sid-resume-identity",
            "algorithm": "sid",
            "recipe_digest": recipe.digest,
            "assets": asset_digests,
            "data": data_identity,
            "parallel_plan_digest": plan.digest,
            "initialization_seed": base_seed,
        }
        checkpoint_state = TrainingState(
            model=checkpoint_model,
            optimizer=(stack.student_optimizer, stack.fake_score_optimizer),
            engine=stack.engine,
            dataloader=sid_loader,
            objective_generator=objective_generator,
            progress=progress,
            identity=identity,
            **stack.checkpoint_state_kwargs(),
        )
        checkpointer = TrainingCheckpointer(destination / "checkpoints")
        resume_artifact = (
            None
            if resume_checkpoint is None
            else checkpointer.load(checkpoint_state, resume_checkpoint)
        )
        session = NativeSIDTrainingSession(
            stack.engine,
            sid_loader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
        )
        return SanaSIDTrainingRun(
            recipe=recipe,
            session=session,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            roles=roles,
            output_dir=destination,
            data_identity=data_identity,
            resume_artifact=resume_artifact,
            distributed_context=distributed_context,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = [
    "SANA_SID_RUN_SCHEMA",
    "SanaSIDTrainingRun",
    "materialize_sana_sid_training_run",
]
