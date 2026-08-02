"""Compose the native training command group without importing training runtimes."""

from __future__ import annotations

import argparse

from .handlers.audit import (
    register_train_audit_rollout_subparser,
    register_train_audit_subparser,
)
from .handlers.cache import register_train_cache_subparser
from .handlers.post_train import register_post_train_subparser
from .handlers.train import register_train_subparser


def register_training_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    register_train_subparser(subparsers)
    register_post_train_subparser(subparsers)
    register_train_cache_subparser(subparsers)
    register_train_audit_subparser(subparsers)
    register_train_audit_rollout_subparser(subparsers)


__all__ = ["register_training_subparser"]
