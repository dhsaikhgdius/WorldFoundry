"""Layerwise CPU offload with asynchronous CUDA prefetch."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import nn

_compiler_disable = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)


@dataclass
class LayerwiseOffloadHandle:
    """Handle returned by ``enable_layerwise_cpu_offload``.

    Holds the registered hook handles so the offload can be undone with
    :meth:`disable` (CC-27); previously the ``RemovableHandle`` objects were
    dropped and a model could never leave offload mode without a rebuild.
    """

    enabled: bool
    layer_count: int
    reason: str = ""
    _model: nn.Module | None = field(default=None, repr=False, compare=False)
    _layer_hooks: list = field(default_factory=list, repr=False, compare=False)

    def disable(self) -> bool:
        """Remove offload hooks and restore real parameters onto each layer.

        Parameters are restored from the CPU master copies (on CPU); move the
        model to the desired device afterwards. Returns True when hooks were
        removed. Handles created for an already-enabled model
        (``reason="already enabled"``) do not own hooks and return False.
        """

        if not self.enabled or self._model is None or not self._layer_hooks:
            self.enabled = False
            return False
        for layer, state, hook_handles in self._layer_hooks:
            for hook_handle in hook_handles:
                hook_handle.remove()
            for name, param in layer.named_parameters(recurse=True):
                cpu_tensor = state.cpu_named_parameters.get(name)
                if cpu_tensor is not None:
                    param.data = cpu_tensor
            state.gpu_named_parameters.clear()
            state.cpu_named_parameters.clear()
            state.next_state = None
            for attribute in ("_worldfoundry_layerwise_cpu_offload", "_worldfoundry_layerwise_cpu_offload_state"):
                layer.__dict__.pop(attribute, None)
        self._model.__dict__.pop("_worldfoundry_layerwise_cpu_offload", None)
        self._layer_hooks.clear()
        self._model = None
        self.enabled = False
        return True


def enable_layerwise_cpu_offload(
    model: nn.Module,
    *,
    layer_container: str | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = True,
) -> LayerwiseOffloadHandle:
    """Attach layerwise CPU offload hooks to the first or named ``ModuleList``.

    The helper keeps parameters on CPU and moves one layer at a time to CUDA.
    The next layer is prefetched on a separate CUDA stream while the current
    layer runs. It is intentionally opt-in and returns a disabled handle on
    non-CUDA systems. Keep the returned handle if you need to switch the
    model back to full-device residency later via ``handle.disable()``.
    """

    if not torch.cuda.is_available():
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason="CUDA is not available")
    if getattr(model, "_worldfoundry_layerwise_cpu_offload", False):
        return LayerwiseOffloadHandle(enabled=True, layer_count=0, reason="already enabled")

    target_device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
    if target_device.type != "cuda":
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason=f"target device is {target_device.type!r}")

    layers = _find_layer_container(model, layer_container)
    if layers is None or len(layers) == 0:
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason="no nn.ModuleList layer container found")

    stream = torch.cuda.Stream(device=target_device)
    states = [_LayerwiseOffloadState(layer, target_device, stream, pin_memory=pin_memory) for layer in layers]
    layer_hooks: list[tuple[nn.Module, _LayerwiseOffloadState, list]] = []
    for index, state in enumerate(states):
        state.next_state = states[(index + 1) % len(states)]
        state.offload_to_cpu()
        layer = layers[index]
        if getattr(layer, "_worldfoundry_layerwise_cpu_offload", False):
            continue
        pre_handle = layer.register_forward_pre_hook(_make_pre_hook(state), with_kwargs=True)
        post_handle = layer.register_forward_hook(_make_post_hook(state), with_kwargs=True)
        layer_hooks.append((layer, state, [pre_handle, post_handle]))
        setattr(layer, "_worldfoundry_layerwise_cpu_offload", True)
        setattr(layer, "_worldfoundry_layerwise_cpu_offload_state", state)
    setattr(model, "_worldfoundry_layerwise_cpu_offload", True)
    return LayerwiseOffloadHandle(
        enabled=True,
        layer_count=len(states),
        _model=model,
        _layer_hooks=layer_hooks,
    )


def layerwise_offload_mutation_scope(module: nn.Module) -> Iterator[None]:
    """Temporarily materialize offloaded parameters for in-place mutations."""

    state = getattr(module, "_worldfoundry_layerwise_cpu_offload_state", None)
    if state is None:
        return nullcontext()
    return state.mutate_params_scope()


class _LayerwiseOffloadState:
    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        async_copy_stream: torch.cuda.Stream,
        *,
        pin_memory: bool,
    ) -> None:
        self.module = module
        self.device = device
        self.async_copy_stream = async_copy_stream
        self.pin_memory = pin_memory
        self.cpu_named_parameters: dict[str, torch.Tensor] = {}
        self.gpu_named_parameters: dict[str, torch.Tensor] = {}
        self.next_state: _LayerwiseOffloadState | None = None

    @_compiler_disable
    def offload_to_cpu(self) -> None:
        for name, param in self.module.named_parameters(recurse=True):
            cpu_tensor = param.detach().to("cpu")
            if self.pin_memory:
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except RuntimeError:
                    pass
            self.cpu_named_parameters[name] = cpu_tensor
            param.data = _tensor_placeholder(param.data, self.device)

    @_compiler_disable
    def wait_and_materialize(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self.async_copy_stream)
        named_parameters = dict(self.module.named_parameters(recurse=True))
        for name, param in named_parameters.items():
            if name not in self.cpu_named_parameters:
                continue
            if name not in self.gpu_named_parameters:
                self.gpu_named_parameters[name] = self.cpu_named_parameters[name].to(self.device, non_blocking=False)
            param.data = self.gpu_named_parameters[name]

    @_compiler_disable
    def prefetch(self) -> None:
        compute_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.async_copy_stream):
            for name, cpu_tensor in self.cpu_named_parameters.items():
                if name in self.gpu_named_parameters:
                    continue
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
                gpu_tensor.record_stream(compute_stream)
                self.gpu_named_parameters[name] = gpu_tensor

    @_compiler_disable
    def release_gpu_params(self) -> None:
        named_parameters = dict(self.module.named_parameters(recurse=True))
        for name, param in named_parameters.items():
            if name not in self.cpu_named_parameters:
                continue
            if name in self.gpu_named_parameters:
                param.data = _tensor_placeholder(param.data, self.device)
                del self.gpu_named_parameters[name]

    @contextmanager
    def mutate_params_scope(self) -> Iterator[None]:
        self.wait_and_materialize()
        try:
            yield
        finally:
            self.cpu_named_parameters.clear()
            self.gpu_named_parameters.clear()
            self.offload_to_cpu()


def _find_layer_container(model: nn.Module, layer_container: str | None) -> nn.ModuleList | None:
    if layer_container:
        current: object = model
        for part in layer_container.split("."):
            current = getattr(current, part)
        return current if isinstance(current, nn.ModuleList) else None
    for module in model.modules():
        for child in module.children():
            if isinstance(child, nn.ModuleList):
                return child
    return None


def _make_pre_hook(state: _LayerwiseOffloadState):
    @_compiler_disable
    def pre_hook(module: nn.Module, args, kwargs):
        state.wait_and_materialize()
        if state.next_state is not None:
            state.next_state.prefetch()
        return args, kwargs

    return pre_hook


def _make_post_hook(state: _LayerwiseOffloadState):
    @_compiler_disable
    def post_hook(module: nn.Module, args, kwargs, output):
        state.release_gpu_params()
        return output

    return post_hook


def _tensor_placeholder(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    shape = (0,) if tensor.ndim <= 0 else (0,) * tensor.ndim
    return torch.empty(shape, dtype=tensor.dtype, device=device)


__all__ = [
    "LayerwiseOffloadHandle",
    "enable_layerwise_cpu_offload",
    "layerwise_offload_mutation_scope",
]
