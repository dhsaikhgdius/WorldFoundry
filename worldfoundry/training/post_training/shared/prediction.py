"""Express WorldFoundry model contracts as shared native flow prediction."""

from __future__ import annotations

from collections.abc import Mapping

from worldfoundry.training.api.contracts import ObjectiveBatch, TrainModelAdapter
from worldfoundry.training.objectives.flow_matching import flow_clean_from_velocity


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("native post-training model prediction requires the 'train-core' extra") from error
    return torch


class NativeFlowPredictionAdapter:
    """Expose a trainable or frozen model adapter as flow prediction.

    SANA/Wan adapters expose ``forward_model`` so frozen teacher/reference roles
    can stay in eval mode.  The fallback is intentionally train-only: silently
    calling ``forward_train`` for a teacher would enable dropout and mutate
    module mode, invalidating DMD and policy replay.
    """

    def __init__(
        self,
        adapter: TrainModelAdapter,
        *,
        autocast_dtype: object | None = None,
        checkpoint_identity: str | None = None,
    ) -> None:
        torch = _require_torch()
        module = getattr(adapter, "trainable_module", None)
        if not isinstance(module, torch.nn.Module):
            raise TypeError("adapter.trainable_module must be an nn.Module")
        if getattr(adapter, "prediction_type", None) != "flow_velocity":
            raise ValueError("native post-training currently requires flow_velocity prediction")
        if autocast_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("autocast_dtype must be float16, bfloat16, or None")
        self.adapter = adapter
        self.module = module
        self.autocast_dtype = autocast_dtype
        inherited_identity = getattr(adapter, "checkpoint_identity", None)
        raw_identity = (
            inherited_identity
            if checkpoint_identity is None
            else checkpoint_identity
        )
        if raw_identity is not None:
            if not isinstance(raw_identity, str) or not raw_identity.strip():
                raise ValueError("checkpoint_identity must be a non-empty string")
            self.checkpoint_identity: str | None = raw_identity.strip()
        else:
            self.checkpoint_identity = None

    def predict_velocity(
        self,
        noisy_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object:
        torch = _require_torch()
        if not torch.is_tensor(noisy_latents) or noisy_latents.ndim < 2:
            raise TypeError("noisy_latents must be a [B,...] torch.Tensor")
        if not noisy_latents.is_floating_point():
            raise TypeError("noisy_latents must have a floating-point dtype")
        if int(noisy_latents.shape[0]) != len(sample_ids):
            raise ValueError("sample_ids length must match noisy_latents")
        if not torch.is_tensor(sigmas):
            sigmas = torch.as_tensor(sigmas, device=noisy_latents.device, dtype=torch.float32)
        sigmas = sigmas.to(device=noisy_latents.device, dtype=torch.float32)
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(int(noisy_latents.shape[0]))
        elif sigmas.numel() == int(noisy_latents.shape[0]):
            sigmas = sigmas.reshape(int(noisy_latents.shape[0]))
        else:
            raise ValueError("sigmas must be scalar or contain one value per sample")
        if branch not in {"positive", "negative"}:
            raise ValueError("branch must be 'positive' or 'negative'")
        # Post-training deliberately keeps trajectory/SDE arithmetic in its
        # configured high-precision dtype.  The denoiser still has to receive
        # its compute dtype explicitly: training adapters disable component-
        # local autocast so the engine owns one precision policy, and Conv3d
        # requires its input and BF16/FP16 bias to agree even when a caller's
        # trajectory is FP32.  The cast is differentiable and mirrors the
        # official Wan training boundary.
        trajectory_dtype = noisy_latents.dtype
        model_input = noisy_latents if self.autocast_dtype is None else noisy_latents.to(dtype=self.autocast_dtype)
        batch = ObjectiveBatch(
            sample_ids=sample_ids,
            model_input=model_input,
            # Post-training consumes the prediction directly; aliasing the input
            # avoids allocating a dummy target the size of a video latent.
            target=model_input,
            sigmas=sigmas,
            timesteps=sigmas,
            conditioning=conditioning,
            metadata={"prediction_type": "flow_velocity"},
        )
        forward_model = getattr(self.adapter, "forward_model", None)
        device_type = noisy_latents.device.type
        autocast_enabled = self.autocast_dtype is not None and device_type in {
            "cpu",
            "cuda",
        }
        if device_type == "cpu" and self.autocast_dtype is torch.float16:
            raise ValueError("CPU post-training does not support float16 autocast")
        with torch.autocast(
            device_type=device_type,
            dtype=self.autocast_dtype,
            enabled=autocast_enabled,
        ):
            if callable(forward_model):
                prediction = forward_model(batch, training=training, branch=branch)
            elif training:
                prediction = self.adapter.forward_train(batch)
            else:
                raise TypeError(
                    "frozen post-training roles require adapter.forward_model(batch, training=False, branch=...)"
                )
        if not torch.is_tensor(prediction) or prediction.shape != noisy_latents.shape:
            raise ValueError("flow adapter prediction must match noisy_latents")
        # Keep all trajectory integration, SDE likelihood, and DMD target math
        # in the caller-owned dtype without severing gradients to the denoiser.
        return prediction.to(dtype=trajectory_dtype)

    def predict_clean(
        self,
        noisy_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object:
        velocity = self.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        return flow_clean_from_velocity(noisy_latents, velocity, sigmas)


class NativeClassifierFreeGuidance:
    """Compose one native policy into the exact Wan CFG prediction.

    Positive and official empty-prompt contexts are concatenated into one
    model call as ``[unconditional, conditional]``.  This keeps rollout and
    replay on the same batch geometry while avoiding any model- or
    repository-specific runtime wrapper.
    """

    def __init__(
        self,
        prediction: NativeFlowPredictionAdapter,
        *,
        guidance_scale: float,
    ) -> None:
        from math import isfinite

        if not isinstance(prediction, NativeFlowPredictionAdapter):
            raise TypeError("native classifier-free guidance requires NativeFlowPredictionAdapter")
        scale = float(guidance_scale)
        if not isfinite(scale) or scale <= 1:
            raise ValueError("classifier-free guidance_scale must be finite and greater than one")
        self.prediction = prediction
        self.module = prediction.module
        self.checkpoint_identity = prediction.checkpoint_identity
        self.guidance_scale = scale

    @staticmethod
    def _batched_conditioning(
        conditioning: Mapping[str, object],
        *,
        batch_size: int,
    ) -> dict[str, object]:
        torch = _require_torch()
        if set(conditioning) != {"context", "negative_context"}:
            raise ValueError("native Wan CFG conditioning requires exactly context and negative_context")
        positive = conditioning["context"]
        negative = conditioning["negative_context"]
        if not isinstance(positive, torch.Tensor) or not isinstance(negative, torch.Tensor):
            raise TypeError("Wan CFG contexts must be torch.Tensor values")
        if positive.ndim < 2 or tuple(positive.shape) != tuple(negative.shape) or int(positive.shape[0]) != batch_size:
            raise ValueError("positive and negative Wan CFG contexts must share shape [B,...]")
        return {"context": torch.cat((negative, positive), dim=0)}

    def predict_velocity(
        self,
        noisy_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object:
        torch = _require_torch()
        if branch != "positive":
            raise ValueError("composed classifier-free guidance exposes only its positive output")
        if not torch.is_tensor(noisy_latents) or int(noisy_latents.shape[0]) != len(sample_ids):
            raise ValueError("CFG noisy_latents and sample_ids must share a batch dimension")
        batch_size = len(sample_ids)
        if not torch.is_tensor(sigmas):
            sigmas = torch.as_tensor(
                sigmas,
                device=noisy_latents.device,
                dtype=torch.float32,
            )
        sigmas = sigmas.to(device=noisy_latents.device, dtype=torch.float32)
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(batch_size)
        elif sigmas.numel() == batch_size:
            sigmas = sigmas.reshape(batch_size)
        else:
            raise ValueError("CFG sigmas must be scalar or contain one value per sample")
        expanded_ids = tuple(f"unconditional::{value}" for value in sample_ids) + tuple(
            f"conditional::{value}" for value in sample_ids
        )
        prediction = self.prediction.predict_velocity(
            torch.cat((noisy_latents, noisy_latents), dim=0),
            torch.cat((sigmas, sigmas), dim=0),
            sample_ids=expanded_ids,
            conditioning=self._batched_conditioning(
                conditioning,
                batch_size=batch_size,
            ),
            training=training,
        )
        unconditional, conditional = prediction.chunk(2, dim=0)
        return unconditional + self.guidance_scale * (conditional - unconditional)

    def predict_clean(
        self,
        noisy_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object:
        velocity = self.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        return flow_clean_from_velocity(noisy_latents, velocity, sigmas)


__all__ = ["NativeClassifierFreeGuidance", "NativeFlowPredictionAdapter"]
