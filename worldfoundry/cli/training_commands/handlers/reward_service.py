"""Registration and execution of the native reward scorer service."""

from __future__ import annotations

import argparse
from pathlib import Path


def _handle_train_reward_service(args: argparse.Namespace) -> int:
    from worldfoundry.training.post_training.rewards.http import (
        NativeRewardService,
        serve_reward_service,
    )
    from worldfoundry.training.post_training.rewards.scorers import (
        build_configured_reward_scorer_registry,
        load_scorer_service_config,
    )

    config = load_scorer_service_config(args.config).override(
        host=args.host,
        port=args.port,
        fail_fast=args.fail_fast,
        device=args.device,
    )
    registry = build_configured_reward_scorer_registry(config)
    service = NativeRewardService(registry, fail_fast=config.fail_fast)
    serve_reward_service(service, host=config.host, port=config.port)
    return 0


def register_train_reward_service_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "train-reward-service",
        help="Serve native media and Agentic reward scorers over HTTP",
        description="Load configured WorldFoundry scorers and run their batched HTTP endpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML or JSON server and scorer configuration.",
    )
    parser.add_argument("--host", help="Override server.host from the config.")
    parser.add_argument("--port", type=int, help="Override server.port from the config.")
    parser.add_argument("--device", help="Override the device for every configured media scorer.")
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether a scorer exception fails the whole request batch.",
    )
    parser.set_defaults(func=_handle_train_reward_service)


__all__ = ["register_train_reward_service_subparser"]
