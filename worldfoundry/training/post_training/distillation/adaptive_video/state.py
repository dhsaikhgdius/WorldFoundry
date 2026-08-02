"""Checkpointable slot-wise regression-loss EMA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import torch

from .math import AdaptiveRegressionObservation

ADAPTIVE_REGRESSION_STATE_SCHEMA = "worldfoundry-adaptive-regression-ema"


class AdaptiveRegressionEMA:
    """Track one regression-loss baseline per student denoising step."""

    def __init__(self, slot_count: int, *, decay: float) -> None:
        if isinstance(slot_count, bool) or int(slot_count) <= 0:
            raise ValueError("slot_count must be a positive integer")
        resolved_decay = float(decay)
        if not isfinite(resolved_decay) or not 0.0 <= resolved_decay < 1.0:
            raise ValueError("decay must be in [0,1)")
        self.slot_count = int(slot_count)
        self.decay = resolved_decay
        self.values = torch.zeros(self.slot_count, dtype=torch.float64)
        self.initialized = torch.zeros(self.slot_count, dtype=torch.bool)
        self.update_counts = torch.zeros(self.slot_count, dtype=torch.int64)

    def commit(self, observations: Sequence[AdaptiveRegressionObservation]) -> None:
        if not observations:
            raise ValueError("an adaptive generator update must contain observations")
        sums = torch.zeros(self.slot_count, dtype=torch.float64)
        counts = torch.zeros(self.slot_count, dtype=torch.int64)
        for observation in observations:
            if not isinstance(observation, AdaptiveRegressionObservation):
                raise TypeError("adaptive observations have an invalid type")
            loss_sums = torch.as_tensor(
                observation.loss_sums,
                dtype=torch.float64,
                device="cpu",
            )
            sample_counts = torch.as_tensor(
                observation.sample_counts,
                dtype=torch.int64,
                device="cpu",
            )
            if loss_sums.shape != (self.slot_count,) or sample_counts.shape != (
                self.slot_count,
            ):
                raise ValueError("adaptive observation shape differs from the schedule")
            if not bool(torch.isfinite(loss_sums).all()) or not bool(
                (sample_counts >= 0).all()
            ):
                raise ValueError("adaptive observation contains invalid statistics")
            sums += loss_sums
            counts += sample_counts
        for slot in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist():
            index = int(slot)
            current = sums[index] / counts[index].to(dtype=torch.float64)
            if bool(self.initialized[index]):
                self.values[index] = (
                    self.decay * self.values[index]
                    + (1.0 - self.decay) * current
                )
            else:
                self.values[index] = current
                self.initialized[index] = True
            self.update_counts[index] += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTIVE_REGRESSION_STATE_SCHEMA,
            "slot_count": self.slot_count,
            "decay": self.decay,
            "values": self.values.clone(),
            "initialized": self.initialized.clone(),
            "update_counts": self.update_counts.clone(),
        }

    def _validated_state(
        self,
        state_dict: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(state_dict, Mapping):
            raise TypeError("adaptive regression state must be a mapping")
        expected = {
            "schema",
            "slot_count",
            "decay",
            "values",
            "initialized",
            "update_counts",
        }
        if set(state_dict) != expected:
            raise ValueError("adaptive regression state fields differ from the active schema")
        if state_dict["schema"] != ADAPTIVE_REGRESSION_STATE_SCHEMA:
            raise ValueError("unsupported adaptive regression state schema")
        if int(state_dict["slot_count"]) != self.slot_count:
            raise ValueError("saved adaptive regression schedule size differs")
        if float(state_dict["decay"]) != self.decay:
            raise ValueError("saved adaptive regression decay differs")
        values = torch.as_tensor(state_dict["values"], dtype=torch.float64, device="cpu")
        initialized = torch.as_tensor(
            state_dict["initialized"],
            dtype=torch.bool,
            device="cpu",
        )
        update_counts = torch.as_tensor(
            state_dict["update_counts"],
            dtype=torch.int64,
            device="cpu",
        )
        expected_shape = (self.slot_count,)
        if (
            values.shape != expected_shape
            or initialized.shape != expected_shape
            or update_counts.shape != expected_shape
        ):
            raise ValueError("adaptive regression tensors differ from the schedule")
        if not bool(torch.isfinite(values).all()) or not bool((update_counts >= 0).all()):
            raise ValueError("adaptive regression state contains invalid values")
        if not torch.equal(initialized, update_counts > 0):
            raise ValueError("adaptive regression initialization and update counts disagree")
        return values.clone(), initialized.clone(), update_counts.clone()

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        values, initialized, update_counts = self._validated_state(state_dict)
        self.values.copy_(values)
        self.initialized.copy_(initialized)
        self.update_counts.copy_(update_counts)


__all__ = ["ADAPTIVE_REGRESSION_STATE_SCHEMA", "AdaptiveRegressionEMA"]
