"""Small VRAM accounting and dynamic model swap helpers."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

cpu = torch.device("cpu")

#: Modules loaded via :func:`load_model_as_complete`. Holds strong references
#: on purpose so :func:`unload_complete_models` can move them back to CPU;
#: callers that drop a model must call ``unload_complete_models`` themselves
#: or the module stays resident until the next unload.
gpu_complete_modules: list[torch.nn.Module] = []


def _default_gpu_device() -> torch.device:
    """Resolve the current CUDA device lazily.

    CC-25: this used to be a module constant evaluated at import time, which
    created a CUDA context on GPU 0 for every importer (including fork-based
    dataloader workers) and froze the device chosen before any
    ``torch.cuda.set_device`` call.
    """

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return cpu


def __getattr__(name: str):
    # Lazy, never-frozen module attribute so ``from ... import gpu`` keeps
    # working without initializing CUDA at import time of this module.
    if name == "gpu":
        return _default_gpu_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs) -> None:
        if "forge_backup_original_class" in module.__dict__:
            # Re-installing would back up the already-hacked class, making
            # uninstall unable to ever restore the real class.
            return
        original_class = module.__class__
        module.__dict__["forge_backup_original_class"] = original_class

        def hacked_get_attr(self, name: str):
            if "_parameters" in self.__dict__:
                parameters = self.__dict__["_parameters"]
                if name in parameters:
                    parameter = parameters[name]
                    if parameter is None:
                        return None
                    if parameter.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(parameter.to(**kwargs), requires_grad=parameter.requires_grad)
                    return parameter.to(**kwargs)
            if "_buffers" in self.__dict__:
                buffers = self.__dict__["_buffers"]
                if name in buffers:
                    return buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type(
            "DynamicSwap_" + original_class.__name__,
            (original_class,),
            {"__getattr__": hacked_get_attr},
        )

    @staticmethod
    def _uninstall_module(module: torch.nn.Module) -> None:
        if "forge_backup_original_class" in module.__dict__:
            module.__class__ = module.__dict__.pop("forge_backup_original_class")

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs) -> None:
        for module in model.modules():
            DynamicSwapInstaller._install_module(module, **kwargs)

    @staticmethod
    def uninstall_model(model: torch.nn.Module) -> None:
        for module in model.modules():
            DynamicSwapInstaller._uninstall_module(module)


def patched_diffusers_current_device(model: torch.nn.Module, target_device: torch.device) -> None:
    if hasattr(model, "scale_shift_table"):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            module.to(target_device)
            return


fake_diffusers_current_device = patched_diffusers_current_device


def get_cuda_free_memory_gb(device=None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = _default_gpu_device()

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats["active_bytes.all.current"]
    bytes_reserved = memory_stats["reserved_bytes.all.current"]
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024**3)


def log_gpu_memory(stage: str, device=None, rank: int = 0) -> None:
    if not torch.cuda.is_available():
        logger.info("[rank %s] [GPU Memory][%s] CUDA unavailable", rank, stage)
        return
    if device is None:
        device = _default_gpu_device()

    free_gb = get_cuda_free_memory_gb(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    used_gb = total_gb - free_gb
    logger.info(
        "[rank %s] [GPU Memory][%s] Used: %.2f GB | Free: %.2f GB | Total: %.2f GB",
        rank,
        stage,
        used_gb,
        free_gb,
        total_gb,
    )


def move_model_to_device_with_memory_preservation(
    model: torch.nn.Module,
    target_device,
    preserved_memory_gb: float = 0,
) -> None:
    logger.info(
        "Moving %s to %s with preserved memory: %s GB", model.__class__.__name__, target_device, preserved_memory_gb
    )
    if preserved_memory_gb < 0:
        raise ValueError("preserved_memory_gb must be non-negative")

    target = _normalize_device(target_device)
    original = _uniform_model_device(model)
    if target.type == "cuda":
        required_gb = _model_transfer_bytes(model, target) / (1024**3)
        available_gb = get_cuda_free_memory_gb(target)
        if available_gb - required_gb < preserved_memory_gb:
            raise RuntimeError(
                f"Insufficient memory to move {model.__class__.__name__} atomically to {target}: "
                f"available={available_gb:.2f} GiB, required~={required_gb:.2f} GiB, "
                f"preserve={preserved_memory_gb:.2f} GiB"
            )

    _move_model_with_rollback(model, target=target, original=original)


def offload_model_from_device_for_memory_preservation(
    model: torch.nn.Module,
    target_device,
    preserved_memory_gb: float = 0,
) -> None:
    logger.info(
        "Offloading %s from %s to preserve memory: %s GB",
        model.__class__.__name__,
        target_device,
        preserved_memory_gb,
    )
    if preserved_memory_gb < 0:
        raise ValueError("preserved_memory_gb must be non-negative")
    if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
        return

    original = _uniform_model_device(model)
    _move_model_with_rollback(model, target=cpu, original=original)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_device(device) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return normalized


def _model_tensors(model: torch.nn.Module):
    for parameter in model.parameters():
        yield parameter
        # Module._apply (and therefore Module.to) migrates existing gradients
        # alongside parameters. Include them in both capacity preflight and
        # device-uniformity validation.
        if parameter.grad is not None:
            yield parameter.grad
    yield from model.buffers()


def _uniform_model_device(model: torch.nn.Module) -> torch.device:
    devices = {tensor.device for tensor in _model_tensors(model)}
    if not devices:
        return cpu
    if len(devices) != 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise RuntimeError(
            f"Refusing to migrate mixed-device {model.__class__.__name__}; "
            f"found tensors on: {rendered}"
        )
    device = next(iter(devices))
    if device.type == "meta":
        raise RuntimeError("Meta-initialized modules require to_empty() and cannot use the VRAM preservation mover")
    return device


def _model_transfer_bytes(model: torch.nn.Module, target: torch.device) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in _model_tensors(model)
        if tensor.device != target
    )


def _move_model_with_rollback(
    model: torch.nn.Module,
    *,
    target: torch.device,
    original: torch.device,
) -> None:
    if target == original:
        return
    try:
        model.to(device=target)
        moved = _uniform_model_device(model)
        if moved != target:
            raise RuntimeError(f"migration ended on {moved}, expected {target}")
    except Exception as move_error:
        try:
            model.to(device=original)
            restored = _uniform_model_device(model)
            if restored != original:
                raise RuntimeError(f"rollback ended on {restored}, expected {original}")
        except Exception as rollback_error:
            raise RuntimeError(
                f"Failed to move {model.__class__.__name__} to {target}; "
                f"rollback to {original} also failed: {rollback_error}"
            ) from move_error
        raise


def unload_complete_models(*models: torch.nn.Module) -> None:
    for model in gpu_complete_modules + list(models):
        model.to(device=cpu)
        logger.info("Unloaded %s as complete.", model.__class__.__name__)

    gpu_complete_modules.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_as_complete(model: torch.nn.Module, target_device, unload: bool = True) -> None:
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    logger.info("Loaded %s to %s as complete.", model.__class__.__name__, target_device)

    gpu_complete_modules.append(model)


__all__ = [
    "DynamicSwapInstaller",
    "cpu",
    "fake_diffusers_current_device",
    "get_cuda_free_memory_gb",
    "gpu",
    "gpu_complete_modules",
    "load_model_as_complete",
    "log_gpu_memory",
    "move_model_to_device_with_memory_preservation",
    "offload_model_from_device_for_memory_preservation",
    "patched_diffusers_current_device",
    "unload_complete_models",
]
