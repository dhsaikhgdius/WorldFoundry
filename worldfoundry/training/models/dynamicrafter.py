"""Native DynamiCrafter image-to-video and interpolation training binding."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch
from worldfoundry.training.objectives.classic_diffusion import (
    ClassicDiffusionConfig,
    ClassicDiffusionObjective,
)

from .lvdm import FramewiseLVDMCodec, freeze_module, latent_sample, module_device_dtype


def _encoded(value: object) -> Tensor:
    if isinstance(value, Tensor):
        return value
    mode = getattr(value, "mode", None)
    if callable(mode):
        result = mode()
        if isinstance(result, Tensor):
            return result
    return latent_sample(value)


def _call_text_encoder(encoder: object, prompts: list[str]) -> Tensor:
    encode = getattr(encoder, "encode", None)
    result = encode(prompts) if callable(encode) else encoder(prompts)  # type: ignore[operator]
    return _encoded(result)


class DynamiCrafterTrainableGraph(nn.Module):
    """Keep the trainable denoiser and image projector in one checkpoint root."""

    def __init__(self, denoiser: nn.Module, image_projector: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.image_projector = image_projector

    def forward(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        *,
        text_context: Tensor,
        image_features: Tensor,
        c_concat: Tensor,
        fps: Tensor,
    ) -> Tensor:
        image_context = self.image_projector(image_features)
        cross_attention = torch.cat((text_context, image_context), dim=1)
        output = self.denoiser(
            noisy,
            timesteps,
            c_concat=[c_concat],
            c_crossattn=[cross_attention],
            fs=fps.long(),
        )
        if isinstance(output, tuple):
            output = output[0]
        sample = getattr(output, "sample", output)
        if not isinstance(sample, Tensor):
            raise TypeError("DynamiCrafter denoiser must return a tensor")
        return sample


class DynamiCrafterTrainAdapter:
    """Own released hybrid conditioning while generic engines own updates."""

    prediction_type = "v_prediction"
    lora_target_preset = None

    def __init__(
        self,
        *,
        denoiser: nn.Module,
        image_projector: nn.Module,
        codec: FramewiseLVDMCodec | None = None,
        text_encoder: object | None = None,
        image_encoder: nn.Module | None = None,
        interpolation: bool = False,
        conditioning_dropout_probability: float = 0.05,
        default_fps: int = 10,
        expected_latent_channels: int = 4,
    ) -> None:
        if not isinstance(denoiser, nn.Module) or not isinstance(image_projector, nn.Module):
            raise TypeError("DynamiCrafter requires nn.Module denoiser and image projector")
        probability = float(conditioning_dropout_probability)
        if not 0.0 <= probability <= 1.0 / 3.0:
            raise ValueError("conditioning dropout probability must be in [0, 1/3]")
        self.denoiser = denoiser
        self.codec = codec
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.trainable_module = DynamiCrafterTrainableGraph(denoiser, image_projector)
        self.interpolation = bool(interpolation)
        self.conditioning_dropout_probability = probability
        self.default_fps = int(default_fps)
        self.expected_latent_channels = int(expected_latent_channels)
        diffusion_model = getattr(denoiser, "diffusion_model", denoiser)
        blocks: list[nn.Module] = []
        for name in ("input_blocks", "output_blocks", "transformer_blocks"):
            values = getattr(diffusion_model, name, ())
            if isinstance(values, (nn.ModuleList, nn.Sequential, list, tuple)):
                blocks.extend(value for value in values if isinstance(value, nn.Module))
        self.fsdp_block_classes = tuple(dict.fromkeys(type(value) for value in blocks))
        frozen: list[nn.Module] = []
        if isinstance(codec, nn.Module):
            frozen.append(codec)
        if isinstance(text_encoder, nn.Module):
            frozen.append(text_encoder)
        if isinstance(image_encoder, nn.Module):
            frozen.append(image_encoder)
        self.frozen_modules = tuple(frozen)
        for module in self.frozen_modules:
            freeze_module(module)

    @classmethod
    def from_latent_visual_diffusion(
        cls,
        model: nn.Module,
        **options: object,
    ) -> "DynamiCrafterTrainAdapter":
        """Bind an already materialized native DynamiCrafter checkpoint."""

        wrapper = getattr(model, "model", None)
        projector = getattr(model, "image_proj_model", None)
        first_stage = getattr(model, "first_stage_model", None)
        text_encoder = getattr(model, "cond_stage_model", None)
        image_encoder = getattr(model, "embedder", None)
        if not all(isinstance(value, nn.Module) for value in (wrapper, projector, first_stage, image_encoder)):
            raise TypeError("DynamiCrafter model is missing its native trainable/conditioning components")
        codec = FramewiseLVDMCodec(
            first_stage,
            scale_factor=float(getattr(model, "scale_factor", 0.18215)),
        )
        options.setdefault("interpolation", bool(getattr(model, "interp_mode", False)))
        options.setdefault(
            "conditioning_dropout_probability",
            float(getattr(model, "uncond_prob", 0.05)),
        )
        return cls(
            denoiser=wrapper,
            image_projector=projector,
            codec=codec,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            **options,
        )

    def _text_conditioning(
        self,
        batch: TrainingBatch,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        context = batch.conditions.get("text_context")
        empty = batch.conditions.get("empty_text_context")
        if context is None or empty is None:
            if self.text_encoder is None:
                raise RuntimeError("DynamiCrafter requires cached text contexts or a text encoder")
            with torch.no_grad():
                context = _call_text_encoder(self.text_encoder, list(batch.prompts))
                empty = _call_text_encoder(self.text_encoder, [""])
        if not isinstance(context, Tensor) or not isinstance(empty, Tensor):
            raise TypeError("DynamiCrafter text contexts must be tensors")
        context = context.detach().to(device=device, dtype=dtype)
        empty = empty.detach().to(device=device, dtype=dtype)
        if int(empty.shape[0]) == 1:
            empty = empty.expand(batch.batch_size, *empty.shape[1:])
        if int(context.shape[0]) != batch.batch_size or int(empty.shape[0]) != batch.batch_size:
            raise ValueError("DynamiCrafter text contexts must match batch size")
        return context, empty

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        device, dtype = module_device_dtype(self.trainable_module)
        cached = batch.conditions.get("clean_latents")
        pixels = batch.pixel_values
        if cached is None:
            if not isinstance(pixels, Tensor) or self.codec is None:
                raise TypeError("DynamiCrafter requires pixels with a codec or cached clean_latents")
            with torch.no_grad():
                clean = self.codec.encode_video(pixels.to(device=device, dtype=dtype))
        else:
            if pixels is not None:
                raise ValueError("DynamiCrafter batch cannot contain both pixels and cached latents")
            if not isinstance(cached, Tensor):
                raise TypeError("clean_latents must be a tensor")
            clean = cached
        if clean.ndim != 5 or int(clean.shape[0]) != batch.batch_size:
            raise ValueError("DynamiCrafter clean latents must be [B,C,T,H,W]")
        if int(clean.shape[1]) != self.expected_latent_channels:
            raise ValueError(f"DynamiCrafter expected {self.expected_latent_channels} latent channels")
        clean = clean.detach().to(device=device, dtype=dtype)
        text_context, empty_text_context = self._text_conditioning(batch, device, dtype)

        image_features = batch.conditions.get("image_features_by_frame")
        zero_image_features = batch.conditions.get("zero_image_features")
        if image_features is not None:
            if not isinstance(image_features, Tensor) or not isinstance(zero_image_features, Tensor):
                raise TypeError("cached image features require image_features_by_frame and zero_image_features tensors")
            if tuple(image_features.shape[:2]) != (batch.batch_size, int(clean.shape[2])):
                raise ValueError("image_features_by_frame must start with [B,T]")
            image_source: dict[str, object] = {
                "image_features_by_frame": image_features.detach().to(device=device, dtype=dtype),
                "zero_image_features": zero_image_features.detach().to(device=device, dtype=dtype),
            }
        else:
            if not isinstance(pixels, Tensor):
                raise RuntimeError("cached DynamiCrafter latents also require cached image features")
            image_source = {"pixel_values": pixels.detach().to(device=device, dtype=dtype)}

        fps = batch.conditions.get("fps", self.default_fps)
        if not isinstance(fps, Tensor):
            fps = torch.full((batch.batch_size,), int(fps), device=device, dtype=torch.long)
        else:
            fps = fps.to(device=device, dtype=torch.long).reshape(batch.batch_size)
        loss_mask = batch.conditions.get("latent_loss_mask")
        if isinstance(loss_mask, Tensor):
            loss_mask = loss_mask.to(device=device, dtype=torch.float32)
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        conditioning = {
            "text_context": text_context,
            "empty_text_context": empty_text_context,
            "fps": fps,
            **image_source,
        }
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean,
            conditioning=conditioning,
            loss_mask=loss_mask,
            sample_weights=sample_weights,
            metadata={
                **dict(batch.metadata),
                "model_family": "dynamicrafter",
                "training_mode": "interpolation" if self.interpolation else "image_to_video",
            },
        )

    def build_objective_conditioning(
        self,
        batch: PreparedBatch,
        timesteps: Tensor,
        clean: Tensor,
        generator: torch.Generator | None,
    ) -> Mapping[str, object]:
        del timesteps
        frame_count = int(clean.shape[2])
        if self.interpolation:
            frame_index = 0
        else:
            frame_index = int(torch.randint(frame_count, (), device=clean.device, generator=generator).item())
        draws = torch.rand(batch.batch_size, device=clean.device, generator=generator)
        probability = self.conditioning_dropout_probability
        drop_text = draws < 2.0 * probability
        drop_image = (draws >= probability) & (draws < 3.0 * probability)
        text = batch.conditioning["text_context"]
        empty = batch.conditioning["empty_text_context"]
        if not isinstance(text, Tensor) or not isinstance(empty, Tensor):
            raise TypeError("DynamiCrafter text conditioning must be tensors")
        text_mask = drop_text.reshape((batch.batch_size,) + (1,) * (text.ndim - 1))
        selected_text = torch.where(text_mask, empty, text)

        if self.interpolation:
            c_concat = torch.zeros_like(clean)
            c_concat[:, :, 0] = clean[:, :, 0]
            c_concat[:, :, -1] = clean[:, :, -1]
        else:
            c_concat = clean[:, :, frame_index : frame_index + 1].expand(-1, -1, frame_count, -1, -1)

        result: dict[str, object] = {
            "text_context": selected_text,
            "c_concat": c_concat,
            "fps": batch.conditioning["fps"],
            "image_drop_mask": drop_image,
            "conditioning_frame_index": frame_index,
        }
        features = batch.conditioning.get("image_features_by_frame")
        if isinstance(features, Tensor):
            result["image_features"] = features[:, frame_index]
            result["zero_image_features"] = batch.conditioning["zero_image_features"]
        else:
            pixels = batch.conditioning.get("pixel_values")
            if not isinstance(pixels, Tensor):
                raise TypeError("DynamiCrafter image source is missing")
            result["conditioning_pixels"] = pixels[:, :, frame_index]
        return result

    def _image_features(self, conditioning: Mapping[str, object]) -> Tensor:
        drop = conditioning["image_drop_mask"]
        if not isinstance(drop, Tensor):
            raise TypeError("image_drop_mask must be a tensor")
        cached = conditioning.get("image_features")
        if isinstance(cached, Tensor):
            zero = conditioning.get("zero_image_features")
            if not isinstance(zero, Tensor):
                raise TypeError("zero_image_features must be a tensor")
            if int(zero.shape[0]) == 1:
                zero = zero.expand(int(cached.shape[0]), *zero.shape[1:])
            mask = drop.reshape((int(cached.shape[0]),) + (1,) * (cached.ndim - 1))
            return torch.where(mask, zero, cached)
        pixels = conditioning.get("conditioning_pixels")
        if not isinstance(pixels, Tensor) or self.image_encoder is None:
            raise RuntimeError("raw DynamiCrafter conditioning requires an image encoder")
        keep = (~drop).to(dtype=pixels.dtype).reshape((int(pixels.shape[0]),) + (1,) * (pixels.ndim - 1))
        with torch.no_grad():
            value = self.image_encoder(pixels * keep)
        return _encoded(value)

    def forward_train(self, batch: ObjectiveBatch) -> Tensor:
        noisy = batch.model_input
        if isinstance(noisy, Mapping) or not isinstance(noisy, Tensor):
            raise TypeError("DynamiCrafter model_input must be one tensor")
        text = batch.conditioning.get("text_context")
        c_concat = batch.conditioning.get("c_concat")
        fps = batch.conditioning.get("fps")
        if not all(isinstance(value, Tensor) for value in (text, c_concat, fps)):
            raise TypeError("DynamiCrafter objective conditioning is incomplete")
        for module in self.frozen_modules:
            module.eval()
        self.trainable_module.train()
        output = self.trainable_module(
            noisy,
            batch.timesteps,
            text_context=text,
            image_features=self._image_features(batch.conditioning),
            c_concat=c_concat,
            fps=fps,
        )
        if output.shape != noisy.shape:
            raise ValueError("DynamiCrafter denoiser output must match noisy latents")
        return output


def dynamicrafter_objective(
    adapter: DynamiCrafterTrainAdapter,
    *,
    dynamic_rescale_final: float = 0.7,
) -> ClassicDiffusionObjective:
    """Build the released v-prediction schedule and hybrid-condition callback."""

    if not isinstance(adapter, DynamiCrafterTrainAdapter):
        raise TypeError("adapter must be DynamiCrafterTrainAdapter")
    return ClassicDiffusionObjective(
        ClassicDiffusionConfig(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="v_prediction",
            loss_type="l2",
            zero_terminal_snr=True,
            dynamic_rescale_final=dynamic_rescale_final,
            dynamic_rescale_transition_steps=400,
        ),
        conditioning_builder=adapter.build_objective_conditioning,
    )


__all__ = [
    "DynamiCrafterTrainAdapter",
    "DynamiCrafterTrainableGraph",
    "dynamicrafter_objective",
]
