"""WorldFoundry-native model graph for ADD feature discrimination."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from math import sqrt
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from worldfoundry.training.optimization import trainable_parameters

from .config import ADDConfig, ADDNoiseSchedule
from .contracts import (
    ADDDecoderAdapter,
    ADDDiscriminatorAdapter,
    ADDDiscriminatorHeadOutput,
    ADDDiscriminatorOutput,
    ADDPredictionAdapter,
)

FeatureLayout = Literal["tokens", "channels-first", "vector"]


class ProjectionFeatureHead(nn.Module):
    """A lightweight patch head with text/image projection conditioning."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        feature_layout: FeatureLayout,
        conditioning_dims: Mapping[str, int],
    ) -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if feature_layout not in {"tokens", "channels-first", "vector"}:
            raise ValueError("feature_layout must be tokens, channels-first, or vector")
        if not isinstance(conditioning_dims, Mapping):
            raise TypeError("conditioning_dims must be a mapping")
        if any(not isinstance(key, str) for key in conditioning_dims):
            raise TypeError("conditioning dimension keys must be strings")
        dims = {key.strip(): value for key, value in conditioning_dims.items()}
        if len(dims) != len(conditioning_dims) or any(not key for key in dims):
            raise ValueError("conditioning dimension keys must be unique and non-empty")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dims.values()):
            raise ValueError("conditioning dimensions must be positive integers")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.feature_layout = feature_layout
        self.conditioning_keys = tuple(dims)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.unconditional_score = nn.Linear(self.hidden_dim, 1)
        self.conditioning_projections = nn.ModuleDict(
            {key: nn.Linear(dimension, self.hidden_dim, bias=False) for key, dimension in dims.items()}
        )
        self.conditioning_dims = dims

    def _tokens(self, features: Tensor) -> Tensor:
        if self.feature_layout == "tokens":
            if features.ndim != 3 or features.shape[-1] != self.feature_dim:
                raise ValueError("token features must have shape [B,N,feature_dim]")
            return features
        if self.feature_layout == "channels-first":
            if features.ndim < 3 or features.shape[1] != self.feature_dim:
                raise ValueError("channels-first features must have shape [B,feature_dim,...]")
            return features.flatten(2).transpose(1, 2)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("vector features must have shape [B,feature_dim]")
        return features[:, None, :]

    def forward(self, features: Tensor, conditioning: Mapping[str, object]) -> Tensor:
        tokens = self._tokens(features)
        if set(conditioning) != set(self.conditioning_keys):
            raise ValueError("discriminator conditioning inventory differs from the feature head")
        hidden = self.feature_projection(tokens.float())
        logits = self.unconditional_score(hidden).squeeze(-1)
        projection = torch.zeros(
            (hidden.shape[0], self.hidden_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        for key in self.conditioning_keys:
            value = conditioning[key]
            if not isinstance(value, Tensor) or value.shape != (
                hidden.shape[0],
                self.conditioning_dims[key],
            ):
                raise ValueError(
                    f"discriminator conditioning {key!r} must have shape [B,{self.conditioning_dims[key]}]"
                )
            projection = projection + self.conditioning_projections[key](
                value.to(device=hidden.device, dtype=hidden.dtype)
            )
        return logits + (hidden * projection[:, None, :]).sum(dim=-1) / sqrt(self.hidden_dim)


class MultiScaleFeatureDiscriminator(nn.Module):
    """Frozen named feature taps and one trainable projection head per tap."""

    def __init__(
        self,
        *,
        feature_network: nn.Module,
        heads: Mapping[tuple[int, str], nn.Module],
        feature_resolutions: tuple[int, ...],
        feature_layers: tuple[str, ...],
        conditioning_keys: tuple[str, ...],
        image_preprocessor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(feature_network, nn.Module):
            raise TypeError("feature_network must be an nn.Module")
        preprocessor = nn.Identity() if image_preprocessor is None else image_preprocessor
        if not isinstance(preprocessor, nn.Module):
            raise TypeError("image_preprocessor must be an nn.Module")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in feature_resolutions):
            raise TypeError("feature_resolutions must contain integers")
        if any(not isinstance(value, str) for value in feature_layers):
            raise TypeError("feature_layers must contain strings")
        resolutions = tuple(feature_resolutions)
        layers = tuple(value.strip() for value in feature_layers)
        keys = tuple((resolution, layer) for resolution in resolutions for layer in layers)
        if not resolutions or any(value <= 0 for value in resolutions) or len(set(resolutions)) != len(resolutions):
            raise ValueError("feature_resolutions must be unique and positive")
        if not layers or any(not value for value in layers) or len(set(layers)) != len(layers):
            raise ValueError("feature_layers must contain unique module paths")
        if any(not isinstance(value, str) for value in conditioning_keys):
            raise TypeError("conditioning_keys must contain strings")
        conditioning = tuple(value.strip() for value in conditioning_keys)
        if any(not value for value in conditioning) or len(set(conditioning)) != len(conditioning):
            raise ValueError("conditioning_keys must contain unique non-empty strings")
        if not isinstance(heads, Mapping):
            raise TypeError("heads must be a mapping")
        normalized_heads: dict[tuple[int, str], nn.Module] = {}
        for key, head in heads.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or isinstance(key[0], bool)
                or not isinstance(key[0], int)
                or not isinstance(key[1], str)
            ):
                raise TypeError("head keys must be (integer resolution, string layer) pairs")
            normalized_key = (key[0], key[1].strip())
            if normalized_key in normalized_heads:
                raise ValueError("head keys must be unique after normalization")
            normalized_heads[normalized_key] = head
        if set(normalized_heads) != set(keys):
            raise ValueError("heads must cover the feature scale/layer Cartesian product exactly")
        if not all(isinstance(normalized_heads[key], nn.Module) for key in keys):
            raise TypeError("every ADD discriminator head must be an nn.Module")
        if any(tuple(getattr(normalized_heads[key], "conditioning_keys", ())) != conditioning for key in keys):
            raise ValueError("every feature head must consume the configured conditioning keys")
        for layer in layers:
            try:
                feature_network.get_submodule(layer)
            except AttributeError as error:
                raise ValueError(f"feature layer {layer!r} does not exist") from error
        if any(parameter.requires_grad for parameter in feature_network.parameters()):
            raise ValueError("ADD feature_network must be frozen before graph construction")
        if any(parameter.requires_grad for parameter in preprocessor.parameters()):
            raise ValueError("ADD image_preprocessor must be frozen before graph construction")
        parameter_ids: list[int] = []
        frozen_graph_parameter_ids = {
            id(parameter) for module in (feature_network, preprocessor) for parameter in module.parameters()
        }
        for key in keys:
            head = normalized_heads[key]
            trainable_parameters(head)
            head_parameter_ids = [id(parameter) for parameter in head.parameters()]
            if frozen_graph_parameter_ids & set(head_parameter_ids):
                raise ValueError("ADD discriminator heads cannot share feature/preprocessor parameters")
            parameter_ids.extend(head_parameter_ids)
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("ADD discriminator heads cannot share parameters")
        self.feature_network = feature_network
        self.image_preprocessor = preprocessor
        self.feature_resolutions = resolutions
        self.feature_layers = layers
        self.conditioning_keys = conditioning
        self.feature_keys = keys
        self.heads = nn.ModuleList(normalized_heads[key] for key in keys)
        self.feature_network.eval()
        self.image_preprocessor.eval()

    def train(self, mode: bool = True) -> MultiScaleFeatureDiscriminator:
        super().train(mode)
        self.feature_network.eval()
        self.image_preprocessor.eval()
        return self

    def _extract(self, images: Tensor, resolution: int, *, track_image_grad: bool) -> tuple[Tensor, ...]:
        captured: dict[str, Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def capture(layer: str):
            def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
                if not isinstance(output, Tensor):
                    raise TypeError(f"feature layer {layer!r} must return one tensor")
                if layer in captured:
                    raise RuntimeError(f"feature layer {layer!r} executed more than once")
                captured[layer] = output

            return hook

        for layer in self.feature_layers:
            handles.append(self.feature_network.get_submodule(layer).register_forward_hook(capture(layer)))
        context = nullcontext() if track_image_grad else torch.no_grad()
        try:
            with context:
                resized = images
                if images.shape[-2:] != (resolution, resolution):
                    resized = F.interpolate(
                        images,
                        size=(resolution, resolution),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                prepared = self.image_preprocessor(resized)
                if not isinstance(prepared, Tensor) or prepared.ndim != 4:
                    raise TypeError("image_preprocessor must return a [B,C,H,W] tensor")
                self.feature_network(prepared)
        finally:
            for handle in handles:
                handle.remove()
        if tuple(captured) != self.feature_layers:
            missing = tuple(layer for layer in self.feature_layers if layer not in captured)
            raise RuntimeError(f"feature network did not execute configured layers: {missing}")
        return tuple(captured[layer] for layer in self.feature_layers)

    def forward(
        self,
        images: Tensor,
        conditioning: Mapping[str, object],
        *,
        track_image_grad: bool,
        require_r1_inputs: bool,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        if not isinstance(images, Tensor) or images.ndim != 4 or images.shape[0] == 0:
            raise TypeError("ADD discriminator images must have shape [B,C,H,W]")
        if track_image_grad and require_r1_inputs:
            raise ValueError("image-gradient and feature-R1 modes are mutually exclusive")
        if set(conditioning) != set(self.conditioning_keys):
            raise ValueError("discriminator conditioning inventory differs from the graph")
        features: list[Tensor] = []
        for resolution in self.feature_resolutions:
            features.extend(self._extract(images, resolution, track_image_grad=track_image_grad))
        if require_r1_inputs:
            features = [value.detach().requires_grad_(True) for value in features]
        elif not track_image_grad:
            features = [value.detach() for value in features]
        logits = [head(value, conditioning) for head, value in zip(self.heads, features, strict=True)]
        return tuple(features), tuple(logits)


class NativeADDDiscriminatorAdapter:
    """Strict runtime adapter for the native multi-scale feature graph."""

    def __init__(self, module: nn.Module, *, checkpoint_identity: str) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("ADD discriminator module must be an nn.Module")
        if not isinstance(checkpoint_identity, str) or not checkpoint_identity.strip():
            raise ValueError("ADD discriminator feature checkpoint_identity must be non-empty")
        identity = checkpoint_identity.strip()
        raw_module = module
        try:
            from torch.nn.parallel import DistributedDataParallel
        except ImportError:
            DistributedDataParallel = ()  # type: ignore[assignment,misc]
        if isinstance(module, DistributedDataParallel):
            raw_module = module.module
        if not isinstance(raw_module, MultiScaleFeatureDiscriminator):
            raise TypeError("ADD discriminator must use MultiScaleFeatureDiscriminator")
        self.module = module
        self.feature_module = raw_module.feature_network
        # The trainable heads are initialized by WorldFoundry.  This identity
        # names the loaded frozen feature network selected by the recipe.
        self.checkpoint_identity = identity
        self.feature_resolutions = raw_module.feature_resolutions
        self.feature_layers = raw_module.feature_layers
        self.conditioning_keys = raw_module.conditioning_keys

    def predict(
        self,
        images: Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        track_image_grad: bool,
        require_r1_inputs: bool,
    ) -> ADDDiscriminatorOutput:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise TypeError("images must be a [B,C,H,W] tensor")
        if len(sample_ids) != images.shape[0]:
            raise ValueError("sample_ids must align with discriminator images")
        result = self.module(
            images,
            conditioning,
            track_image_grad=track_image_grad,
            require_r1_inputs=require_r1_inputs,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("native ADD discriminator must return (features, logits)")
        features, logits = result
        expected_keys = tuple(
            (resolution, layer) for resolution in self.feature_resolutions for layer in self.feature_layers
        )
        if not isinstance(features, tuple) or not isinstance(logits, tuple):
            raise TypeError("native ADD feature and logit inventories must be tuples")
        if len(features) != len(expected_keys) or len(logits) != len(expected_keys):
            raise ValueError("native ADD discriminator returned an incomplete feature inventory")
        outputs: list[ADDDiscriminatorHeadOutput] = []
        for key, feature, logit in zip(expected_keys, features, logits, strict=True):
            if not isinstance(feature, Tensor) or feature.ndim < 2 or feature.shape[0] != images.shape[0]:
                raise ValueError(f"ADD feature {key!r} must preserve the batch dimension")
            if not isinstance(logit, Tensor) or logit.ndim < 1 or logit.shape[0] != images.shape[0]:
                raise ValueError(f"ADD logits {key!r} must preserve the batch dimension")
            if not bool(torch.isfinite(feature).all()) or not bool(torch.isfinite(logit).all()):
                raise FloatingPointError(f"ADD discriminator output {key!r} is non-finite")
            if require_r1_inputs and (not feature.requires_grad or not feature.is_leaf):
                raise RuntimeError("R1 feature inputs must be detached gradient-requiring leaves")
            outputs.append(
                ADDDiscriminatorHeadOutput(
                    resolution=key[0],
                    layer=key[1],
                    features=feature,
                    logits=logit,
                )
            )
        return ADDDiscriminatorOutput(heads=tuple(outputs))


class ADDTrainableRoles(nn.Module):
    """One DCP-visible module containing exactly the two trainable ADD roles."""

    def __init__(self, student: nn.Module, discriminator: nn.Module) -> None:
        super().__init__()
        if not isinstance(student, nn.Module) or not isinstance(discriminator, nn.Module):
            raise TypeError("ADD trainable roles must be nn.Module values")
        if student is discriminator:
            raise ValueError("ADD student and discriminator must be distinct")
        self.student = student
        self.discriminator = discriminator


def audit_add_model_graph(
    *,
    student: ADDPredictionAdapter,
    teacher: ADDPredictionAdapter,
    decoder: ADDDecoderAdapter,
    discriminator: ADDDiscriminatorAdapter,
    student_schedule: ADDNoiseSchedule,
    teacher_schedule: ADDNoiseSchedule,
    config: ADDConfig,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """Fail closed unless every paper role and schedule is executable."""

    if not isinstance(student, ADDPredictionAdapter):
        raise TypeError("student must implement ADDPredictionAdapter")
    if not isinstance(teacher, ADDPredictionAdapter):
        raise TypeError("teacher must implement ADDPredictionAdapter")
    if not isinstance(decoder, ADDDecoderAdapter):
        raise TypeError("decoder must implement ADDDecoderAdapter")
    if not isinstance(discriminator, ADDDiscriminatorAdapter):
        raise TypeError("discriminator must implement ADDDiscriminatorAdapter")
    if not isinstance(config, ADDConfig):
        raise TypeError("config must be ADDConfig")
    if not isinstance(student_schedule, ADDNoiseSchedule):
        raise TypeError("student_schedule must be ADDNoiseSchedule")
    if not isinstance(teacher_schedule, ADDNoiseSchedule):
        raise TypeError("teacher_schedule must be ADDNoiseSchedule")
    modules = (
        student.module,
        teacher.module,
        decoder.module,
        discriminator.module,
        discriminator.feature_module,
    )
    if not all(isinstance(module, nn.Module) for module in modules):
        raise TypeError("every ADD adapter must expose its executable nn.Module")
    student_module, teacher_module, decoder_module, discriminator_module, feature_module = modules
    if len({id(student_module), id(teacher_module), id(decoder_module), id(feature_module)}) != 4:
        raise ValueError("ADD student, teacher, decoder, and feature network must be distinct")
    for role, module in (
        ("teacher", teacher_module),
        ("decoder", decoder_module),
        ("feature network", feature_module),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(f"ADD {role} must be frozen")
        module.eval()
    trainable_parameters(student_module)
    discriminator_parameters = trainable_parameters(discriminator_module)
    role_inventories = {
        "student": {id(parameter) for parameter in student_module.parameters()},
        "teacher": {id(parameter) for parameter in teacher_module.parameters()},
        "decoder": {id(parameter) for parameter in decoder_module.parameters()},
        "feature network": {id(parameter) for parameter in feature_module.parameters()},
        "discriminator heads": {id(parameter) for parameter in discriminator_parameters},
    }
    role_names = tuple(role_inventories)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            if role_inventories[left] & role_inventories[right]:
                raise ValueError(f"ADD roles {left!r} and {right!r} cannot share parameters")
    if tuple(discriminator.feature_resolutions) != config.feature_resolutions:
        raise ValueError("ADD discriminator resolutions differ from the active config")
    if tuple(discriminator.feature_layers) != config.feature_layers:
        raise ValueError("ADD discriminator layers differ from the active config")
    if tuple(discriminator.conditioning_keys) != config.discriminator_conditioning_keys:
        raise ValueError("ADD discriminator conditioning keys differ from the active config")
    if config.student_timesteps[-1] >= student_schedule.num_timesteps:
        raise ValueError("ADD student timesteps exceed the student noise schedule")
    if student_schedule.training_loss_weights is not None:
        raise ValueError("student training_loss_weights are unused by ADD")
    if config.student_timesteps[-1] != student_schedule.num_timesteps - 1:
        raise ValueError("the final ADD student timestep must address the schedule terminal")
    if student_schedule.alpha_cumprods[config.student_timesteps[-1]] != 0.0:
        raise ValueError("the final ADD student timestep must have exactly zero terminal SNR")
    if config.teacher_timestep_max >= teacher_schedule.num_timesteps:
        raise ValueError("ADD teacher timestep range exceeds the teacher noise schedule")
    selected_teacher_alphas = teacher_schedule.alpha_cumprods[
        config.teacher_timestep_min : config.teacher_timestep_max + 1
    ]
    if any(not 0.0 < value < 1.0 for value in selected_teacher_alphas):
        raise ValueError("ADD teacher timestep range requires non-zero signal and noise power")
    if config.distillation_weighting == "exponential":
        if teacher_schedule.training_loss_weights is not None:
            raise ValueError("teacher training_loss_weights are unused by exponential ADD")
    elif teacher_schedule.training_loss_weights is None:
        raise ValueError("SDS ADD requires teacher training_loss_weights")
    return modules


__all__ = [
    "ADDTrainableRoles",
    "FeatureLayout",
    "MultiScaleFeatureDiscriminator",
    "NativeADDDiscriminatorAdapter",
    "ProjectionFeatureHead",
    "audit_add_model_graph",
]
