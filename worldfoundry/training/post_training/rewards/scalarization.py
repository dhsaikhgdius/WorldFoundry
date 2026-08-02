"""Content-addressed vector reward scalarization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from worldfoundry.core.io.integrity import canonical_sha256


@dataclass(frozen=True, slots=True)
class RewardScalarizationResult:
    scalar_rewards: object
    normalized_components: Mapping[str, object]
    valid_mask: object
    scalarizer_digest: str


class WeightedRewardScalarizer:
    """Frozen-calibration weighted sum with explicit invalid-result policy."""

    schema = "worldfoundry-reward-scalarizer"

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        calibration_mean: Mapping[str, float] | None = None,
        calibration_std: Mapping[str, float] | None = None,
        normalization_epsilon: float = 0.0,
        invalid_policy: str = "reject",
    ) -> None:
        resolved_weights = {str(key): float(value) for key, value in weights.items()}
        if not resolved_weights or any(not key.strip() for key in resolved_weights):
            raise ValueError("reward scalarizer weights must be a non-empty mapping")
        if any(not isfinite(value) for value in resolved_weights.values()) or not any(
            value != 0 for value in resolved_weights.values()
        ):
            raise ValueError("reward weights must be finite and not all zero")
        mean = (
            {key: 0.0 for key in resolved_weights}
            if calibration_mean is None
            else {str(key): float(value) for key, value in calibration_mean.items()}
        )
        std = (
            {key: 1.0 for key in resolved_weights}
            if calibration_std is None
            else {str(key): float(value) for key, value in calibration_std.items()}
        )
        if set(mean) != set(resolved_weights) or set(std) != set(resolved_weights):
            raise ValueError("calibration keys must exactly match reward weights")
        if any(not isfinite(value) for value in mean.values()):
            raise ValueError("calibration means must be finite")
        if any(not isfinite(value) or value <= 0 for value in std.values()):
            raise ValueError("calibration standard deviations must be finite and positive")
        epsilon = float(normalization_epsilon)
        if not isfinite(epsilon) or epsilon < 0:
            raise ValueError("normalization_epsilon must be finite and non-negative")
        if invalid_policy not in {"reject", "zero"}:
            raise ValueError("invalid_policy must be 'reject' or 'zero'")
        self.weights = MappingProxyType(resolved_weights)
        self.calibration_mean = MappingProxyType(mean)
        self.calibration_std = MappingProxyType(std)
        self.normalization_epsilon = epsilon
        self.invalid_policy = invalid_policy

    @property
    def digest(self) -> str:
        return canonical_sha256(self.state_dict())

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "weights": dict(self.weights),
            "calibration_mean": dict(self.calibration_mean),
            "calibration_std": dict(self.calibration_std),
            "normalization_epsilon": self.normalization_epsilon,
            "invalid_policy": self.invalid_policy,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schema",
            "weights",
            "calibration_mean",
            "calibration_std",
            "normalization_epsilon",
            "invalid_policy",
        }:
            raise ValueError("reward scalarizer state fields differ from the active schema")
        if state_dict["schema"] != self.schema:
            raise ValueError(f"unsupported reward scalarizer schema: {state_dict['schema']!r}")
        candidate = WeightedRewardScalarizer(
            state_dict["weights"],  # type: ignore[arg-type]
            calibration_mean=state_dict["calibration_mean"],  # type: ignore[arg-type]
            calibration_std=state_dict["calibration_std"],  # type: ignore[arg-type]
            normalization_epsilon=float(state_dict["normalization_epsilon"]),
            invalid_policy=str(state_dict["invalid_policy"]),
        )
        if candidate.state_dict() != self.state_dict():
            raise ValueError("saved reward scalarizer differs from the active run")

    def scalarize(
        self,
        values: Mapping[str, object],
        *,
        valid: Mapping[str, object] | None = None,
    ) -> RewardScalarizationResult:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("tensor reward scalarization requires the 'train-core' extra") from error
        if set(values) != set(self.weights):
            raise ValueError("reward component keys must exactly match scalarizer weights")
        tensors = dict(values)
        if not all(torch.is_tensor(value) and value.ndim == 1 for value in tensors.values()):
            raise TypeError("reward components must be one-dimensional torch.Tensor values")
        shapes = {tuple(value.shape) for value in tensors.values()}
        if len(shapes) != 1:
            raise ValueError("reward component tensors must share shape [B]")
        reference = next(iter(tensors.values()))
        if valid is None:
            valid_tensors = {key: torch.isfinite(value) for key, value in tensors.items()}
        else:
            if set(valid) != set(self.weights):
                raise ValueError("reward validity keys must exactly match scalarizer weights")
            valid_tensors = {}
            for key, value in valid.items():
                if not torch.is_tensor(value) or value.shape != reference.shape or value.dtype is not torch.bool:
                    raise TypeError("reward validity values must be bool tensors with shape [B]")
                valid_tensors[key] = value.to(device=reference.device)
        joint_valid = torch.ones_like(reference, dtype=torch.bool)
        for key, value in tensors.items():
            joint_valid &= valid_tensors[key] & torch.isfinite(value)
        if self.invalid_policy == "reject" and not bool(joint_valid.all()):
            invalid = int((~joint_valid).sum().item())
            raise ValueError(f"reward scalarization received {invalid} invalid samples")

        scalar = torch.zeros_like(reference, dtype=torch.float32)
        normalized: dict[str, object] = {}
        for key, value in tensors.items():
            component = (value.to(device=reference.device, dtype=torch.float32) - self.calibration_mean[key]) / (
                self.calibration_std[key] + self.normalization_epsilon
            )
            if self.invalid_policy == "zero":
                component = torch.where(joint_valid, component, torch.zeros_like(component))
            normalized[key] = component
            scalar = scalar + self.weights[key] * component
        return RewardScalarizationResult(
            scalar_rewards=scalar,
            normalized_components=MappingProxyType(normalized),
            valid_mask=joint_valid,
            scalarizer_digest=self.digest,
        )


__all__ = ["RewardScalarizationResult", "WeightedRewardScalarizer"]
