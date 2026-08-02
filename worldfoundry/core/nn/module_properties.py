"""Small framework-independent properties for tensor-owning modules."""

from __future__ import annotations

import torch


class ModuleDeviceDtypeMixin:
    """Expose the device and dtype of a module's first tensor.

    Native model adapters use this instead of inheriting an external pipeline
    framework solely for its ``.device`` and ``.dtype`` conveniences.
    """

    def _reference_tensor(self) -> torch.Tensor:
        reference = next(getattr(self, "parameters")(), None)
        if reference is None:
            reference = next(getattr(self, "buffers")(), None)
        if reference is None:
            raise RuntimeError(f"{type(self).__name__} does not own a parameter or buffer")
        return reference

    @property
    def device(self) -> torch.device:
        return self._reference_tensor().device

    @property
    def dtype(self) -> torch.dtype:
        return self._reference_tensor().dtype


__all__ = ["ModuleDeviceDtypeMixin"]
