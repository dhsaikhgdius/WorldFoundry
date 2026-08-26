"""Rank-aware logging helpers for distributed runtime code."""

from __future__ import annotations

import logging

import torch


class DistributedLogger:
    _logger: logging.Logger | None = None

    @classmethod
    def get_logger(cls, *, level: int = logging.INFO) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger("worldfoundry_distributed")
            cls._logger.setLevel(level)
            cls._logger.propagate = False
            cls._logger.handlers.clear()
            formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            cls._logger.addHandler(handler)
        return cls._logger


distributed_logger = DistributedLogger.get_logger()


def _adopt_unified_pipeline_if_configured() -> None:
    """Join the unified logging pipeline when it was configured first.

    CM-01: ``configure_logging`` deliberately does not import this module (it
    would drag ``torch`` into lightweight CLI/catalog processes); it only
    reparents the singleton when this module is already loaded. Cover the
    opposite import order here: when the pipeline is already configured,
    reparent immediately so ``print_rank_0`` & co. flow through the root
    handlers regardless of which side loaded first.
    """
    from worldfoundry.core.logging_setup import is_configured

    if not is_configured():
        return
    distributed_logger.handlers.clear()
    distributed_logger.propagate = True
    distributed_logger.setLevel(logging.getLogger().level)


_adopt_unified_pipeline_if_configured()

try:  # Keep the shared distributed helpers usable in minimal model environments.
    from loguru import logger as _backend_logger
except ImportError:  # pragma: no cover - exercised in isolated runtime environments.
    _backend_logger = distributed_logger


def print_per_rank(message: object) -> None:
    """Emit one informational log record from every calling rank."""
    distributed_logger.info(message)


def print_rank_0(message: object) -> None:
    """Emit one informational record on rank zero, or in a non-distributed process."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            distributed_logger.info(message)
    else:
        distributed_logger.info(message)


class _RankAwareLog:
    """Loguru-compatible facade for model code that supports rank-zero filtering."""

    logger = _backend_logger

    @staticmethod
    def _enabled(rank0_only: bool) -> bool:
        return not rank0_only or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def _write(self, level: str, message, *args, rank0_only: bool = False, **kwargs) -> None:
        if self._enabled(rank0_only):
            method = getattr(_backend_logger, "info" if level == "success" else level)
            if _backend_logger is distributed_logger:
                # Loguru call sites commonly use ``{}`` placeholders and keyword
                # options that the stdlib logger does not accept.
                try:
                    message = str(message).format(*args)
                    args = ()
                except (IndexError, KeyError, ValueError):
                    pass
                kwargs = {key: value for key, value in kwargs.items() if key in {"exc_info", "stack_info", "stacklevel", "extra"}}
            method(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self._write("debug", message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._write("info", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._write("warning", message, *args, **kwargs)

    warn = warning

    def error(self, message, *args, **kwargs):
        self._write("error", message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._write("critical", message, *args, **kwargs)

    def success(self, message, *args, **kwargs):
        self._write("success", message, *args, **kwargs)


log = _RankAwareLog()


__all__ = ["DistributedLogger", "distributed_logger", "log", "print_per_rank", "print_rank_0"]
