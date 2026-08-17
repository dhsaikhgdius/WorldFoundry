"""Wan denoiser adapter for the native diffusion execution contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

from worldfoundry.core.model_loading import hash_state_dict_keys

from ...components import BuildPurpose, ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.wan.model import RMSNorm, WanModel

WAN21_T2V_1P3B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "eps": 1e-6,
}

WAN21_T2V_14B_CONFIG = {
    **WAN21_T2V_1P3B_CONFIG,
    "dim": 5120,
    "ffn_dim": 13824,
    "num_heads": 40,
    "num_layers": 40,
}

WAN22_T2V_A14B_CONFIG = dict(WAN21_T2V_14B_CONFIG)

WAN21_I2V_14B_CONFIG = {
    **WAN21_T2V_14B_CONFIG,
    "has_image_input": True,
    "in_dim": 36,
}

# Some Wan2.1 research checkpoints (for example WoW 1.3B) train the
# first-frame VAE condition without adding the CLIP cross-attention branch.
# This is still the canonical Wan graph: only its explicit conditioning roles
# differ from the released 14B I2V checkpoint.
WAN21_VAE_I2V_1P3B_CONFIG = {
    **WAN21_T2V_1P3B_CONFIG,
    "in_dim": 36,
    "require_vae_embedding": True,
    "require_clip_embedding": False,
}

# Wan2.2 A14B I2V conditions the latent stream with the VAE mask/latent
# tensor but does not own the Wan2.1 CLIP image branch.  Keeping that split in
# one canonical graph is required by Fun-Control and FantasyWorld releases.
WAN22_I2V_A14B_CONFIG = {
    **WAN21_T2V_14B_CONFIG,
    "has_image_input": False,
    "in_dim": 36,
    "require_vae_embedding": True,
    "require_clip_embedding": False,
}

WAN22_TI2V_5B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "per_token_timestep": True,
}

_WAN_DENOISER_OPTION_KEYS = frozenset({"peft_adapter_path", "weight_dtype"})


def _read_wan_peft_config(adapter_path: Path) -> dict[str, object]:
    config_path = adapter_path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Wan PEFT adapter config: {config_path}") from error
    if not isinstance(config, dict):
        raise TypeError("Wan PEFT adapter_config.json must contain one JSON object")
    if config.get("peft_type") != "LORA":
        raise ValueError(f"unsupported Wan PEFT adapter type: {config.get('peft_type')!r}")
    return config


def _validated_wan_peft_adapter(
    context: ComponentBuildContext,
    value: object,
) -> Path:
    if context.purpose is BuildPurpose.TRAINING:
        raise ValueError("Wan inference PEFT adapters cannot be supplied to a training component build")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError("Wan peft_adapter_path must be a non-empty local filesystem path")

    adapter_path = Path(value).expanduser()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Wan PEFT adapter directory does not exist: {adapter_path}")
    config = _read_wan_peft_config(adapter_path)
    base_model_id = config.get("base_model_name_or_path")
    if base_model_id not in (None, "") and str(base_model_id) != context.model_id:
        raise ValueError(
            "Wan PEFT adapter base model differs from the selected model: "
            f"adapter={base_model_id!r}, selected={context.model_id!r}"
        )
    return adapter_path


def _merge_wan_peft_adapter(model: torch.nn.Module, adapter_path: Path) -> None:
    from worldfoundry.training.tuning import WAN_ATTENTION, audit_lora_targets, merge_peft_adapter

    config = _read_wan_peft_config(adapter_path)
    target_modules = config.get("target_modules")
    if not target_modules or not isinstance(target_modules, (str, list)):
        raise ValueError("Wan PEFT adapter must declare target_modules")
    target_audit = audit_lora_targets(model, WAN_ATTENTION)
    auto_mapping = config.get("auto_mapping")
    if isinstance(auto_mapping, Mapping):
        base_class = auto_mapping.get("base_model_class")
        if base_class not in (None, "", type(model).__name__):
            raise ValueError(
                "Wan PEFT adapter base architecture differs from the loaded model: "
                f"adapter={base_class!r}, loaded={type(model).__name__!r}"
            )
    try:
        from peft import PeftModel
    except ModuleNotFoundError as error:
        raise RuntimeError("loading a Wan LoRA adapter requires the PEFT dependency") from error
    loaded = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    targeted = set(getattr(loaded, "targeted_module_names", ()))
    if targeted != set(target_audit.module_names):
        raise ValueError("Wan PEFT adapter targets are incompatible with the loaded Wan graph")
    merged = merge_peft_adapter(loaded)
    if merged is not model:
        raise RuntimeError("PEFT merge did not return the loaded Wan base model")

SKYREELS_V2_DF_1P3B_CONFIG = {
    **WAN21_T2V_1P3B_CONFIG,
    "inject_sample_info": True,
}

SKYREELS_V3_R2V_14B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-6,
}


WAN_CIVITAI_CONFIGS_BY_HASH = {
    "9269f8db9040a9d860eaca435be61814": WAN21_T2V_1P3B_CONFIG,
    "aafcfd9672c3a2456dc46e1cb6e52c70": WAN21_T2V_14B_CONFIG,
    "6bfcfb3b342cb286ce886889d519a77e": WAN21_I2V_14B_CONFIG,
    "6d6ccde6845b95ad9114ab993d917893": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 36,
    },
    "349723183fc063b2bfc10bb2835cf677": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
    },
    "efa44cddf936c70abd0ea28b6cbe946c": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
    },
    "3ef3b1f8e1dab83d5b71fd7b617f859f": {
        **WAN21_I2V_14B_CONFIG,
        "has_image_pos_emb": True,
    },
    "70ddad9d3a133785da5ea371aae09504": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
        "has_ref_conv": True,
    },
    "26bde73488a92e64cc20b0a7485b9e5b": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
        "has_ref_conv": True,
    },
    "ac6a5aa74f4a0aab6f64eb9a72f19901": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 32,
        "add_control_adapter": True,
        "in_dim_control_adapter": 24,
    },
    "b61c605c2adbd23124d152ed28e049ae": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 32,
        "add_control_adapter": True,
        "in_dim_control_adapter": 24,
    },
}


def infer_native_wan_transformer_config(
    state_dict: Mapping[str, object],
) -> dict[str, object]:
    """Infer the canonical Wan graph from checkpoint tensor structure.

    Released research checkpoints often keep native Wan parameter names but
    have a new key hash after changing input conditioning.  The architecture
    is nevertheless fully described by the patch/head/block tensor shapes and
    optional module keys, so those checkpoints should not need a model-specific
    backend or an ever-growing hash table.
    """

    def tensor(name: str) -> torch.Tensor:
        value = state_dict.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Wan checkpoint is missing tensor {name!r}")
        return value

    patch = tensor("patch_embedding.weight")
    head = tensor("head.head.weight")
    text = tensor("text_embedding.0.weight")
    time = tensor("time_embedding.0.weight")
    ffn = tensor("blocks.0.ffn.0.weight")
    if patch.ndim != 5:
        raise ValueError(f"Wan patch embedding must be Conv3d-shaped, got {tuple(patch.shape)}")

    block_ids = {
        int(name.split(".", 2)[1])
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor)
        and name.startswith("blocks.")
        and len(name.split(".", 2)) == 3
        and name.split(".", 2)[1].isdigit()
    }
    if not block_ids or block_ids != set(range(max(block_ids) + 1)):
        raise ValueError("Wan checkpoint blocks must form one contiguous zero-based sequence")

    patch_size = tuple(int(size) for size in patch.shape[2:])
    patch_volume = math.prod(patch_size)
    if head.shape[0] % patch_volume:
        raise ValueError(
            f"Wan output projection {tuple(head.shape)} is incompatible with patch size {patch_size}"
        )
    dim = int(patch.shape[0])
    if dim % 128:
        raise ValueError(f"Wan hidden dimension {dim} is not divisible by the canonical head width 128")

    has_image_input = any(
        name.startswith("img_emb.") or ".cross_attn.k_img." in name
        for name in state_dict
    )
    has_control_adapter = any(name.startswith("control_adapter.") for name in state_dict)
    config: dict[str, object] = {
        "has_image_input": has_image_input,
        "patch_size": patch_size,
        "in_dim": int(patch.shape[1]),
        "dim": dim,
        "ffn_dim": int(ffn.shape[0]),
        "freq_dim": int(time.shape[1]),
        "text_dim": int(text.shape[1]),
        "out_dim": int(head.shape[0] // patch_volume),
        "num_heads": dim // 128,
        "num_layers": max(block_ids) + 1,
        "eps": 1e-6,
        "has_image_pos_emb": "img_emb.emb_pos" in state_dict,
        "has_ref_conv": any(name.startswith("ref_conv.") for name in state_dict),
        "add_control_adapter": has_control_adapter,
        "require_vae_embedding": int(patch.shape[1]) != int(head.shape[0] // patch_volume),
        "require_clip_embedding": has_image_input,
        "inject_sample_info": "fps_embedding.weight" in state_dict,
    }
    if has_control_adapter:
        adapter_weight = tensor("control_adapter.conv.weight")
        config["in_dim_control_adapter"] = int(adapter_weight.shape[1] // 64)
    return config


def convert_diffusers_wan_transformer_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Map Diffusers or already-native Wan names onto the native graph.

    Official Wan releases use ``diffusion_pytorch_model`` filenames for both
    layouts.  Stable-Video-Infinity stages the native Wan2.1 shards under that
    name, so filename-based loading may legitimately invoke this converter on
    native keys.
    """

    replacements = (
        (".attn1.to_out.0.", ".self_attn.o."),
        (".attn1.to_q.", ".self_attn.q."),
        (".attn1.to_k.", ".self_attn.k."),
        (".attn1.to_v.", ".self_attn.v."),
        (".attn1.norm_q.", ".self_attn.norm_q."),
        (".attn1.norm_k.", ".self_attn.norm_k."),
        (".attn2.to_out.0.", ".cross_attn.o."),
        (".attn2.to_q.", ".cross_attn.q."),
        (".attn2.to_k.", ".cross_attn.k."),
        (".attn2.to_v.", ".cross_attn.v."),
        (".attn2.norm_q.", ".cross_attn.norm_q."),
        (".attn2.norm_k.", ".cross_attn.norm_k."),
        (".ffn.net.0.proj.", ".ffn.0."),
        (".ffn.net.2.", ".ffn.2."),
        (".norm2.", ".norm3."),
        (".scale_shift_table", ".modulation"),
    )
    roots = {
        "condition_embedder.text_embedder.linear_1.": "text_embedding.0.",
        "condition_embedder.text_embedder.linear_2.": "text_embedding.2.",
        "condition_embedder.time_embedder.linear_1.": "time_embedding.0.",
        "condition_embedder.time_embedder.linear_2.": "time_embedding.2.",
        "condition_embedder.time_proj.": "time_projection.1.",
        "proj_out.": "head.head.",
        "scale_shift_table": "head.modulation",
    }
    native_roots = (
        "patch_embedding.",
        "text_embedding.",
        "time_embedding.",
        "time_projection.",
        "img_emb.",
        "head.",
    )
    converted: dict[str, object] = {}
    for source, value in state_dict.items():
        target = source if source.startswith(native_roots) else roots.get(source)
        if target is None:
            for prefix, replacement in roots.items():
                if source.startswith(prefix):
                    target = replacement + source[len(prefix) :]
                    break
        if target is None and source.startswith("blocks."):
            target = source
            for old, new in replacements:
                target = target.replace(old, new)
        if target is None:
            raise KeyError(f"unsupported Diffusers Wan transformer parameter: {source}")
        if target in converted:
            raise KeyError(f"Wan transformer conversion produced duplicate parameter: {target}")
        converted[target] = value
    return converted


class WanModelStateDictConverter:
    """Detect released Wan checkpoint layouts for native model construction."""

    def from_diffusers(self, state_dict):
        converted = convert_diffusers_wan_transformer_state_dict(state_dict)
        config = WAN21_T2V_14B_CONFIG if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b" else {}
        return converted, dict(config)

    def from_civitai(self, state_dict):
        raw_hash = hash_state_dict_keys(state_dict)
        filtered = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("vace")
            and not (name.startswith("control") and not name.startswith("control_adapter"))
        }
        checkpoint_hash = raw_hash if raw_hash in {
            "ac6a5aa74f4a0aab6f64eb9a72f19901",
            "b61c605c2adbd23124d152ed28e049ae",
        } else hash_state_dict_keys(filtered)
        config = WAN_CIVITAI_CONFIGS_BY_HASH.get(checkpoint_hash)
        if config is None:
            config = infer_native_wan_transformer_config(filtered)
        return filtered, dict(config)


class WanDenoiser:
    """Expose a checkpoint-compatible Wan DiT through ``Denoiser``."""

    def __init__(
        self,
        model: WanModel,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
        reference_condition_key: str | None = None,
        manage_autocast: bool = True,
    ) -> None:
        if not isinstance(manage_autocast, bool):
            raise TypeError("Wan manage_autocast must be a bool")
        self.model = model
        self.compute_dtype = compute_dtype
        self.reference_condition_key = reference_condition_key
        self.manage_autocast = manage_autocast

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        try:
            context = model_input.conditioning["context"]
        except KeyError as error:
            raise KeyError("Wan denoising requires a 'context' conditioning tensor") from error
        if not isinstance(context, torch.Tensor):
            raise TypeError("Wan 'context' conditioning must be a tensor")

        latents = model_input.latents
        timestep = model_input.timestep.to(
            device=latents.device,
            dtype=torch.float32,
        ).reshape(-1)
        if timestep.numel() == 1 and model_input.latents.shape[0] != 1:
            timestep = timestep.expand(model_input.latents.shape[0])
        if timestep.numel() != model_input.latents.shape[0]:
            raise ValueError("Wan timestep must be scalar or have one value per latent sample")
        if self.model.per_token_timestep:
            denoise_mask = model_input.conditioning.get("denoise_mask")
            if not isinstance(denoise_mask, torch.Tensor):
                raise TypeError("per-token Wan denoising requires a tensor denoise_mask")
            denoise_mask = denoise_mask.to(device=latents.device, dtype=latents.dtype)
            if denoise_mask.ndim != 5 or denoise_mask.shape[0] != latents.shape[0]:
                raise ValueError("Wan denoise_mask must have shape [B,C,T,H,W]")
            patch_height, patch_width = self.model.patch_size[1:]
            token_mask = denoise_mask[:, 0, :, ::patch_height, ::patch_width].flatten(1)
            timestep = timestep[:, None] * token_mask
        model_kwargs = {}
        use_gradient_checkpointing = model_input.conditioning.get(
            "use_gradient_checkpointing",
            False,
        )
        use_gradient_checkpointing_offload = model_input.conditioning.get(
            "use_gradient_checkpointing_offload",
            False,
        )
        if not isinstance(use_gradient_checkpointing, bool):
            raise TypeError("Wan use_gradient_checkpointing must be a bool")
        if not isinstance(use_gradient_checkpointing_offload, bool):
            raise TypeError("Wan use_gradient_checkpointing_offload must be a bool")
        model_kwargs.update(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        if self.model.inject_sample_info:
            fps = int(model_input.conditioning.get("fps", 24))
            fps_id = 0 if fps == 16 else 1
            model_kwargs["fps"] = torch.full(
                (model_input.latents.shape[0],),
                fps_id,
                device=model_input.latents.device,
                dtype=torch.long,
            )
        model_latents = latents
        if self.reference_condition_key is not None:
            reference = model_input.conditioning.get(self.reference_condition_key)
            if not isinstance(reference, torch.Tensor):
                raise TypeError(
                    f"Wan reference conditioning {self.reference_condition_key!r} must be a tensor"
                )
            reference = reference.to(device=latents.device, dtype=latents.dtype)
            if bool(model_input.conditioning.get("drop_secondary_condition", False)):
                reference = torch.zeros_like(reference)
            if reference.shape[:2] != latents.shape[:2] or reference.shape[-2:] != latents.shape[-2:]:
                raise ValueError(
                    "Wan reference latents must match generated batch/channel/spatial geometry: "
                    f"{tuple(reference.shape)} vs {tuple(latents.shape)}"
                )
            model_latents = torch.cat((latents, reference), dim=2)

        if self.model.has_image_input:
            condition_latents = model_input.conditioning.get("condition_latents")
            clip_feature = model_input.conditioning.get("clip_feature")
            if not isinstance(condition_latents, torch.Tensor):
                raise TypeError("Wan image-to-video denoising requires tensor condition_latents")
            if not isinstance(clip_feature, torch.Tensor):
                raise TypeError("Wan image-to-video denoising requires tensor clip_feature")
            model_kwargs.update(
                y=condition_latents.to(device=latents.device, dtype=latents.dtype),
                clip_feature=clip_feature.to(device=latents.device, dtype=latents.dtype),
            )

        autocast_enabled = self.compute_dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(
            device_type=latents.device.type,
            dtype=self.compute_dtype,
            enabled=self.manage_autocast and autocast_enabled,
        ):
            sample = self.model(
                x=model_latents,
                timestep=timestep,
                context=context,
                **model_kwargs,
            )
        sample = sample[:, :, : latents.shape[2]]
        return DenoiserOutput(sample=sample)


def build_wan21_t2v_1p3b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the Wan2.1 T2V 1.3B DiT with shared core placement policy."""

    return _build_wan_denoiser(context, config=WAN21_T2V_1P3B_CONFIG)


def build_skyreels_v2_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the SkyReels-V2 1.3B graph on the shared Wan architecture."""

    return _build_wan_denoiser(context, config=SKYREELS_V2_DF_1P3B_CONFIG)


def build_wan21_t2v_14b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the Wan2.1 T2V 14B checkpoint on the shared Wan graph."""

    return _build_wan_denoiser(context, config=WAN21_T2V_14B_CONFIG)


def build_wan22_t2v_a14b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load either released Wan2.2 A14B expert on the native Wan graph."""

    return _build_wan_denoiser(context, config=WAN22_T2V_A14B_CONFIG)


def build_wan21_i2v_14b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the Wan2.1 I2V 14B checkpoint on the shared Wan graph."""

    return _build_wan_denoiser(context, config=WAN21_I2V_14B_CONFIG)


def build_wan22_ti2v_5b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load Wan2.2 TI2V 5B on the shared Wan graph with token timesteps."""

    return _build_wan_denoiser(context, config=WAN22_TI2V_5B_CONFIG)


def build_skyreels_v3_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load SkyReels-V3 R2V weights into the shared native Wan graph."""

    return _build_wan_denoiser(
        context,
        config=SKYREELS_V3_R2V_14B_CONFIG,
        state_dict_converter=convert_diffusers_wan_transformer_state_dict,
        reference_condition_key="reference_latents",
    )


def _build_wan_denoiser(
    context: ComponentBuildContext,
    *,
    config: dict[str, object],
    state_dict_converter=None,
    reference_condition_key: str | None = None,
) -> WanDenoiser:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    unknown_options = sorted(set(context.component_options) - _WAN_DENOISER_OPTION_KEYS)
    if unknown_options:
        raise ValueError(f"unsupported Wan denoiser options: {unknown_options}")
    weight_dtype = context.component_options.get("weight_dtype", torch.float32)
    if not isinstance(weight_dtype, torch.dtype):
        raise TypeError(f"Wan denoiser weight dtype must be a torch.dtype, got {weight_dtype!r}")
    adapter_value = context.component_options.get("peft_adapter_path")
    adapter_path = (
        None
        if adapter_value is None
        else _validated_wan_peft_adapter(context, adapter_value)
    )

    def merge_training_adapter(model: torch.nn.Module) -> None:
        if adapter_path is None:
            return
        _merge_wan_peft_adapter(model, adapter_path)

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=WanModel,
            config=config,
            state_dict_converter=state_dict_converter,
            vram_module_map={
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            layer_container="blocks",
            post_load_hook=merge_training_adapter if adapter_path is not None else None,
        ),
        context.require_checkpoint("weights"),
        replace(context.policy, dtype=weight_dtype),
    )
    if not isinstance(model, WanModel):
        raise TypeError(f"expected WanModel, got {type(model).__name__}")
    return WanDenoiser(
        model,
        compute_dtype=context.policy.dtype,
        reference_condition_key=reference_condition_key,
        manage_autocast=context.purpose is not BuildPurpose.TRAINING,
    )


__all__ = [
    "SKYREELS_V2_DF_1P3B_CONFIG",
    "SKYREELS_V3_R2V_14B_CONFIG",
    "WAN21_I2V_14B_CONFIG",
    "WAN21_VAE_I2V_1P3B_CONFIG",
    "WAN22_I2V_A14B_CONFIG",
    "WAN21_T2V_1P3B_CONFIG",
    "WAN21_T2V_14B_CONFIG",
    "WAN22_T2V_A14B_CONFIG",
    "WAN22_TI2V_5B_CONFIG",
    "WanDenoiser",
    "WanModelStateDictConverter",
    "WAN_CIVITAI_CONFIGS_BY_HASH",
    "infer_native_wan_transformer_config",
    "build_skyreels_v2_denoiser",
    "build_skyreels_v3_denoiser",
    "build_wan21_i2v_14b_denoiser",
    "convert_diffusers_wan_transformer_state_dict",
    "build_wan21_t2v_14b_denoiser",
    "build_wan22_t2v_a14b_denoiser",
    "build_wan21_t2v_1p3b_denoiser",
    "build_wan22_ti2v_5b_denoiser",
]
