"""Native declarative recipes for Wan model variants."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.wan import build_wan_video_decoder, build_wan_video_vae38_decoder
from ..models.denoisers.wan import (
    build_wan21_i2v_14b_denoiser,
    build_wan21_t2v_1p3b_denoiser,
    build_wan21_t2v_14b_denoiser,
    build_wan22_ti2v_5b_denoiser,
)
from ..models.denoisers.wan_vace import build_wan21_vace_14b_denoiser
from ..models.encoders.wan import build_wan_image_text_conditioner, build_wan_text_conditioner
from ..models.initializers.wan import (
    build_wan_i2v_latent_initializer,
    build_wan_t2v_latent_initializer,
    build_wan_ti2v_latent_initializer,
    build_wan_vace_latent_initializer,
)
from ..schedulers import build_wan_flow_unipc_scheduler
from .spec import NativeDiffusionRecipe

WAN21_T2V_1P3B_MODEL_ID = "wan2.1-t2v-1.3b"
WAN21_T2V_14B_MODEL_ID = "wan2.1-t2v-14b"
WAN21_I2V_14B_480P_MODEL_ID = "wan2.1-i2v-14b-480p"
WAN21_I2V_14B_720P_MODEL_ID = "wan2.1-i2v-14b-720p"
WAN22_TI2V_5B_MODEL_ID = "wan2.2-ti2v-5b"
WAN21_VACE_14B_MODEL_ID = "wan2.1-vace"

WAN21_T2V_1P3B_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
WAN21_T2V_14B_REPO_ID = "Wan-AI/Wan2.1-T2V-14B"
WAN21_I2V_14B_480P_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN21_I2V_14B_720P_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-720P"
WAN22_TI2V_5B_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN21_VACE_14B_REPO_ID = "Wan-AI/Wan2.1-VACE-14B"

WAN21_T2V_1P3B_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
WAN21_T2V_14B_REVISION = "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"
WAN21_I2V_14B_480P_REVISION = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
WAN21_I2V_14B_720P_REVISION = "8823af45fcc58a8aa999a54b04be9abc7d2aac98"
WAN22_TI2V_5B_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
WAN21_VACE_14B_REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"

# Architecture/training semantics were audited against the official Wan
# source.  Model assets are independently pinned to the immutable Hub commit
# above so a README-only source change cannot silently alter checkpoint
# identity.
WAN21_UPSTREAM_SOURCE_REVISION = "9737cba9c1c3c4d04b33fcad41c111989865d315"
WAN21_T2V_1P3B_FILE_SHA256 = {
    "diffusion_pytorch_model.safetensors": (
        "96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f"
    ),
    "models_t5_umt5-xxl-enc-bf16.pth": (
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
    ),
    "Wan2.1_VAE.pth": (
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
    ),
    "google/umt5-xxl/spiece.model": (
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"
    ),
    "google/umt5-xxl/tokenizer.json": (
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
    ),
}
WAN21_T2V_1P3B_FILE_SIZE_BYTES = {
    "diffusion_pytorch_model.safetensors": 5_676_070_424,
    "models_t5_umt5-xxl-enc-bf16.pth": 11_361_920_418,
    "Wan2.1_VAE.pth": 507_609_880,
    "google/umt5-xxl/special_tokens_map.json": 6_623,
    "google/umt5-xxl/spiece.model": 4_548_313,
    "google/umt5-xxl/tokenizer.json": 16_837_417,
    "google/umt5-xxl/tokenizer_config.json": 61_728,
}

WAN_TOKENIZER_FILES = (
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


def _keys() -> tuple[ComponentKey, ...]:
    return (
        ComponentKey(ComponentKind.DENOISER),
        ComponentKey(ComponentKind.CONDITIONER),
        ComponentKey(ComponentKind.LATENT_INITIALIZER),
        ComponentKey(ComponentKind.SCHEDULER),
        ComponentKey(ComponentKind.LATENT_ENCODER, "codec"),
    )


def _execution(*, image_to_video: bool) -> ExecutionSpec:
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    bindings = {
        "denoiser": denoiser,
        "conditioner": conditioner,
        "latent_initializer": initializer,
        "scheduler": scheduler,
        "decoder": codec,
    }
    if image_to_video:
        bindings["latent_encoder"] = codec
    return ExecutionSpec(bindings=bindings)


def _checkpoints(
    repo_id: str,
    revision: str,
    *,
    sharded_dit: bool,
    image_to_video: bool,
    file_sha256: Mapping[str, str] | None = None,
    file_size_bytes: Mapping[str, int] | None = None,
) -> dict[str, CheckpointSpec]:
    integrity = dict(file_sha256 or {})
    sizes = dict(file_size_bytes or {})
    metadata = {
        "license": "Apache-2.0",
        "repository_revision": revision,
        "upstream_source_revision": WAN21_UPSTREAM_SOURCE_REVISION,
    }
    dit_file = (
        "diffusion_pytorch_model.safetensors.index.json"
        if sharded_dit
        else "diffusion_pytorch_model.safetensors"
    )
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(dit_file,),
            allow_patterns=("diffusion_pytorch_model*",),
            metadata=metadata,
            file_sha256=(
                {dit_file: integrity[dit_file]}
                if dit_file in integrity
                else {}
            ),
            file_size_bytes=(
                {dit_file: sizes[dit_file]}
                if dit_file in sizes
                else {}
            ),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
            metadata=metadata,
            file_sha256=(
                {
                    "models_t5_umt5-xxl-enc-bf16.pth": integrity[
                        "models_t5_umt5-xxl-enc-bf16.pth"
                    ]
                }
                if "models_t5_umt5-xxl-enc-bf16.pth" in integrity
                else {}
            ),
            file_size_bytes=(
                {
                    "models_t5_umt5-xxl-enc-bf16.pth": sizes[
                        "models_t5_umt5-xxl-enc-bf16.pth"
                    ]
                }
                if "models_t5_umt5-xxl-enc-bf16.pth" in sizes
                else {}
            ),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
            metadata=metadata,
            file_sha256={
                name: integrity[name]
                for name in WAN_TOKENIZER_FILES
                if name in integrity
            },
            file_size_bytes={
                name: sizes[name]
                for name in WAN_TOKENIZER_FILES
                if name in sizes
            },
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("Wan2.1_VAE.pth",),
            metadata=metadata,
            file_sha256=(
                {"Wan2.1_VAE.pth": integrity["Wan2.1_VAE.pth"]}
                if "Wan2.1_VAE.pth" in integrity
                else {}
            ),
            file_size_bytes=(
                {"Wan2.1_VAE.pth": sizes["Wan2.1_VAE.pth"]}
                if "Wan2.1_VAE.pth" in sizes
                else {}
            ),
        ),
    }
    if image_to_video:
        checkpoints["image-encoder"] = CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",),
        )
    return checkpoints


def _wan21_recipe(
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    denoiser_factory: Callable,
    parameter_scale: str,
    shift: float,
    aliases: tuple[str, ...],
    image_to_video: bool,
    resolution: str | None = None,
) -> NativeDiffusionRecipe:
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = _checkpoints(
        repo_id,
        revision,
        sharded_dit=parameter_scale == "14B",
        image_to_video=image_to_video,
        file_sha256=(
            WAN21_T2V_1P3B_FILE_SHA256
            if model_id == WAN21_T2V_1P3B_MODEL_ID
            else None
        ),
        file_size_bytes=(
            WAN21_T2V_1P3B_FILE_SIZE_BYTES
            if model_id == WAN21_T2V_1P3B_MODEL_ID
            else None
        ),
    )
    conditioner_checkpoints = {
        "weights": "text-encoder",
        "tokenizer": "tokenizer",
    }
    if image_to_video:
        conditioner_checkpoints["image_weights"] = "image-encoder"
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, denoiser_factory, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_wan_image_text_conditioner if image_to_video else build_wan_text_conditioner,
                conditioner_checkpoints,
            ),
            ComponentSpec(
                initializer,
                build_wan_i2v_latent_initializer
                if image_to_video
                else build_wan_t2v_latent_initializer,
            ),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": shift},
            ),
            ComponentSpec(codec, build_wan_video_decoder, {"weights": "vae"}),
        ),
        execution=_execution(image_to_video=image_to_video),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "image-to-video" if image_to_video else "text-to-video",
                "classifier-free-guidance",
            }
        ),
        options={
            "latent_channels": 16,
            "spatial_compression": 8,
            "temporal_compression": 4,
        },
        metadata={
            "architecture": "wan2.1",
            "parameter_scale": parameter_scale,
            "resolution": resolution,
            "native_inference": True,
            "output_layout": "BCTHW",
            "license": "Apache-2.0",
            "upstream_source_revision": WAN21_UPSTREAM_SOURCE_REVISION,
        },
    )


def wan21_t2v_1p3b_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 T2V 1.3B recipe."""

    return _wan21_recipe(
        model_id=WAN21_T2V_1P3B_MODEL_ID,
        repo_id=WAN21_T2V_1P3B_REPO_ID,
        revision=WAN21_T2V_1P3B_REVISION,
        denoiser_factory=build_wan21_t2v_1p3b_denoiser,
        parameter_scale="1.3B",
        shift=8.0,
        aliases=(
            "wan2.1",
            "wan-2.1",
            "wan2p1",
            "wan2.1-t2v",
            "wan2p1-t2v",
            "wan21-t2v-1.3b",
            WAN21_T2V_1P3B_REPO_ID,
        ),
        image_to_video=False,
    )


def wan21_t2v_14b_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 T2V 14B recipe."""

    return _wan21_recipe(
        model_id=WAN21_T2V_14B_MODEL_ID,
        repo_id=WAN21_T2V_14B_REPO_ID,
        revision=WAN21_T2V_14B_REVISION,
        denoiser_factory=build_wan21_t2v_14b_denoiser,
        parameter_scale="14B",
        shift=5.0,
        aliases=("wan21-t2v-14b", WAN21_T2V_14B_REPO_ID),
        image_to_video=False,
    )


def wan21_i2v_14b_480p_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 I2V 14B 480P recipe."""

    return _wan21_recipe(
        model_id=WAN21_I2V_14B_480P_MODEL_ID,
        repo_id=WAN21_I2V_14B_480P_REPO_ID,
        revision=WAN21_I2V_14B_480P_REVISION,
        denoiser_factory=build_wan21_i2v_14b_denoiser,
        parameter_scale="14B",
        shift=3.0,
        aliases=("wan2.1-i2v", "wan2p1-i2v", "wan21-i2v-480p", WAN21_I2V_14B_480P_REPO_ID),
        image_to_video=True,
        resolution="480P",
    )


def wan21_i2v_14b_720p_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 I2V 14B 720P recipe."""

    return _wan21_recipe(
        model_id=WAN21_I2V_14B_720P_MODEL_ID,
        repo_id=WAN21_I2V_14B_720P_REPO_ID,
        revision=WAN21_I2V_14B_720P_REVISION,
        denoiser_factory=build_wan21_i2v_14b_denoiser,
        parameter_scale="14B",
        shift=5.0,
        aliases=("wan21-i2v-720p", WAN21_I2V_14B_720P_REPO_ID),
        image_to_video=True,
        resolution="720P",
    )


def wan21_vace_14b_recipe() -> NativeDiffusionRecipe:
    """Return Wan2.1 VACE 14B on the framework-owned sampling loop."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=("diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("diffusion_pytorch_model*", "config.json"),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=("Wan2.1_VAE.pth",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=WAN21_VACE_14B_MODEL_ID,
        aliases=("wan-vace", "wan2.1-vace-14b", WAN21_VACE_14B_REPO_ID),
        components=(
            ComponentSpec(denoiser, build_wan21_vace_14b_denoiser, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(initializer, build_wan_vace_latent_initializer),
            ComponentSpec(scheduler, build_wan_flow_unipc_scheduler, options={"shift": 5.0}),
            ComponentSpec(codec, build_wan_video_decoder, {"weights": "vae"}),
        ),
        execution=_execution(image_to_video=True),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "controlled-video-generation",
                "image-to-video",
                "reference-to-video",
                "video-to-video",
                "classifier-free-guidance",
            }
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": "wan2.1-vace",
            "parameter_scale": "14B",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


def wan22_ti2v_5b_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.2 5B recipe with optional first-frame locking."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=("diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("diffusion_pytorch_model*",),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=("Wan2.2_VAE.pth",),
        ),
    }
    bindings = dict(_execution(image_to_video=True).bindings)
    return NativeDiffusionRecipe(
        model_id=WAN22_TI2V_5B_MODEL_ID,
        aliases=(
            "wan2.2",
            "wan-2.2",
            "wan2p2",
            "wan2.2-ti2v-5b-1280x704-121f",
            WAN22_TI2V_5B_REPO_ID,
        ),
        components=(
            ComponentSpec(denoiser, build_wan22_ti2v_5b_denoiser, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(initializer, build_wan_ti2v_latent_initializer),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": 5.0},
            ),
            ComponentSpec(codec, build_wan_video_vae38_decoder, {"weights": "vae"}),
        ),
        execution=ExecutionSpec(strategy="masked-latent", bindings=bindings),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {"text-to-video", "image-to-video", "classifier-free-guidance", "per-token-timestep"}
        ),
        options={
            "latent_channels": 48,
            "spatial_compression": 16,
            "temporal_compression": 4,
        },
        metadata={
            "architecture": "wan2.2-ti2v",
            "parameter_scale": "5B",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


__all__ = [
    "WAN21_I2V_14B_480P_MODEL_ID",
    "WAN21_I2V_14B_480P_REPO_ID",
    "WAN21_I2V_14B_480P_REVISION",
    "WAN21_I2V_14B_720P_MODEL_ID",
    "WAN21_I2V_14B_720P_REPO_ID",
    "WAN21_I2V_14B_720P_REVISION",
    "WAN21_T2V_1P3B_MODEL_ID",
    "WAN21_T2V_1P3B_REPO_ID",
    "WAN21_T2V_1P3B_REVISION",
    "WAN21_T2V_1P3B_FILE_SHA256",
    "WAN21_T2V_14B_MODEL_ID",
    "WAN21_T2V_14B_REPO_ID",
    "WAN21_T2V_14B_REVISION",
    "WAN22_TI2V_5B_MODEL_ID",
    "WAN22_TI2V_5B_REPO_ID",
    "WAN22_TI2V_5B_REVISION",
    "WAN21_VACE_14B_MODEL_ID",
    "WAN21_VACE_14B_REPO_ID",
    "WAN21_VACE_14B_REVISION",
    "WAN_TOKENIZER_FILES",
    "WAN21_UPSTREAM_SOURCE_REVISION",
    "wan21_i2v_14b_480p_recipe",
    "wan21_i2v_14b_720p_recipe",
    "wan21_t2v_1p3b_recipe",
    "wan21_t2v_14b_recipe",
    "wan22_ti2v_5b_recipe",
    "wan21_vace_14b_recipe",
]
