"""Native Cosmos Predict 2.5 2B and 14B recipes."""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.cosmos2p5 import build_cosmos25_video_codec
from ..models.denoisers.cosmos2p5 import (
    build_cosmos25_2b_denoiser,
    build_cosmos25_14b_denoiser,
    build_cosmos25_transfer_2b_denoiser,
)
from ..models.encoders.cosmos2p5 import build_cosmos25_prompt_conditioner
from ..schedulers import build_wan_flow_unipc_scheduler
from .spec import NativeDiffusionRecipe

COSMOS25_2B_MODEL_ID = "cosmos-predict2.5-2b"
COSMOS25_14B_MODEL_ID = "cosmos-predict2.5-14b"
COSMOS25_TRANSFER_2B_MODEL_ID = "cosmos-transfer2.5-2b-controlled-video"
COSMOS25_2B_REPO_ID = "nvidia/Cosmos-Predict2.5-2B"
COSMOS25_14B_REPO_ID = "nvidia/Cosmos-Predict2.5-14B"
COSMOS25_TRANSFER_2B_REPO_ID = "nvidia/Cosmos-Transfer2.5-2B"
COSMOS25_REASON_REPO_ID = "nvidia/Cosmos-Reason1-7B"
COSMOS25_2B_REVISION = "f176dc95b4a70f53ce01c4b302851595e7322b00"
COSMOS25_14B_REVISION = "71ebf3e8af30ecfe440bf0481115975fcc052b46"
COSMOS25_REASON_REVISION = "375e24000b24baed78f4618d3dd779e47cd96323"
COSMOS25_TRANSFER_2B_REVISION = "bd963eabcfc2d61dc4ea365cacf41d45ac480aa5"

_REASON_TOKENIZER_FILES = (
    "chat_template.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _recipe(
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    weight_file: str,
    denoiser_factory,
    aliases: tuple[str, ...],
    capabilities: frozenset[str] | None = None,
    architecture: str = "cosmos-predict2.5-minimal-v1-lvg-dit",
) -> NativeDiffusionRecipe:
    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    codec = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(weight_file,),
            allow_patterns=(weight_file,),
        ),
        "vae": CheckpointSpec(
            repo_id=COSMOS25_2B_REPO_ID,
            revision=COSMOS25_2B_REVISION,
            files=("tokenizer.pth",),
            allow_patterns=("tokenizer.pth",),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=COSMOS25_REASON_REPO_ID,
            revision=COSMOS25_REASON_REVISION,
            files=("model.safetensors.index.json",),
            allow_patterns=("model.safetensors.index.json", "model*.safetensors"),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=COSMOS25_REASON_REPO_ID,
            revision=COSMOS25_REASON_REVISION,
            files=_REASON_TOKENIZER_FILES,
            allow_patterns=_REASON_TOKENIZER_FILES,
        ),
    }
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, denoiser_factory, {"weights": "transformer"}),
            ComponentSpec(
                conditioner,
                build_cosmos25_prompt_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(codec, build_cosmos25_video_codec, {"weights": "vae"}),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": 5.0, "use_karras_sigma": True},
            ),
        ),
        execution=ExecutionSpec(
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": codec,
                "scheduler": scheduler,
                "decoder": codec,
            },
            options={"guidance_mode": "positive"},
        ),
        checkpoints=checkpoints,
        capabilities=capabilities
        or frozenset(
            {
                "text-to-world",
                "image-to-world",
                "video-to-world",
                "classifier-free-guidance",
            }
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": architecture,
            "native_inference": True,
            "output_layout": "BCTHW",
            "upstream_revision": "a2c298b0a3df3778b973fe65e9e58877b292d8a7",
        },
    )


def cosmos25_2b_recipe() -> NativeDiffusionRecipe:
    return _recipe(
        model_id=COSMOS25_2B_MODEL_ID,
        repo_id=COSMOS25_2B_REPO_ID,
        revision=COSMOS25_2B_REVISION,
        weight_file="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        denoiser_factory=build_cosmos25_2b_denoiser,
        aliases=("cosmos-predict2.5", "cosmos-predict2p5", COSMOS25_2B_REPO_ID),
    )


def cosmos25_14b_recipe() -> NativeDiffusionRecipe:
    return _recipe(
        model_id=COSMOS25_14B_MODEL_ID,
        repo_id=COSMOS25_14B_REPO_ID,
        revision=COSMOS25_14B_REVISION,
        weight_file="base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt",
        denoiser_factory=build_cosmos25_14b_denoiser,
        aliases=(COSMOS25_14B_REPO_ID,),
    )


def cosmos25_transfer_2b_recipe() -> NativeDiffusionRecipe:
    return _recipe(
        model_id=COSMOS25_TRANSFER_2B_MODEL_ID,
        repo_id=COSMOS25_TRANSFER_2B_REPO_ID,
        revision=COSMOS25_TRANSFER_2B_REVISION,
        weight_file="general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
        denoiser_factory=build_cosmos25_transfer_2b_denoiser,
        aliases=(
            "cosmos-transfer-2.5",
            "cosmos-transfer2.5",
            "cosmos-transfer2p5",
            "cosmos-transfer-2.5-2b",
            COSMOS25_TRANSFER_2B_REPO_ID,
        ),
        capabilities=frozenset(
            {
                "controlled-video",
                "edge-to-world",
                "image-to-world",
                "classifier-free-guidance",
            }
        ),
        architecture="cosmos-transfer2.5-minimal-v4-lvg-vace-dit",
    )


__all__ = ["cosmos25_2b_recipe", "cosmos25_14b_recipe", "cosmos25_transfer_2b_recipe"]
