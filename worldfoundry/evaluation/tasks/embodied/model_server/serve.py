"""WebSocket model server for embodied policy adapters."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from worldfoundry.evaluation.tasks.embodied.policy_adapter import (
    EmbodiedPolicyAdapter,
    build_policy_adapter,
    normalize_action_payload,
)
from worldfoundry.studio.serving.auth import is_loopback_host, token_matches

from .protocol import (
    PROTOCOL_VERSION,
    Message,
    MessageType,
    assert_compatible_protocol_version,
    hello_payload,
    pack_message,
    unpack_message,
)

logger = logging.getLogger(__name__)

EMBODIED_SERVER_TOKEN_ENV = "WF_EMBODIED_SERVER_TOKEN"
DEFAULT_HOST = "127.0.0.1"
# WebSocket-level keepalive; previously disabled (ping_interval=None), which hides
# half-open connections during long embodied rollouts. Set env to 0 to disable.
DEFAULT_PING_INTERVAL_S = float(os.getenv("WF_EMBODIED_WS_PING_INTERVAL_S", "20") or "20")
DEFAULT_MAX_MESSAGE_BYTES = int(os.getenv("WF_EMBODIED_WS_MAX_SIZE", str(64 * 1024 * 1024)))


def _configured_embodied_token() -> str:
    return os.getenv(EMBODIED_SERVER_TOKEN_ENV, "").strip()


def require_embodied_auth_token_for_host(host: str) -> str:
    """Fail closed when binding a non-loopback host without a shared token."""

    if is_loopback_host(host):
        return ""
    token = _configured_embodied_token()
    if not token:
        raise SystemExit(
            f"embodied model_server refuses to bind non-loopback host {host!r} without authentication. "
            f"Set {EMBODIED_SERVER_TOKEN_ENV} to a shared secret (clients must send "
            "'Authorization: Bearer <token>' during the WebSocket handshake), "
            f"or bind --host {DEFAULT_HOST}."
        )
    return token


def _token_from_headers(headers: Any) -> str | None:
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if not callable(get):
        return None
    authorization = get("Authorization") or get("authorization")
    if isinstance(authorization, bytes):
        authorization = authorization.decode("utf-8", errors="replace")
    if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def _serializable_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in spec.items():
        to_dict = getattr(value, "to_dict", None)
        payload[str(key)] = to_dict() if callable(to_dict) else value
    return payload


async def _maybe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _handle_connection(ws: Any, adapter: EmbodiedPolicyAdapter) -> None:
    async for raw in ws:
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        try:
            message = unpack_message(raw)
        except Exception as exc:
            await ws.send(pack_message(Message(MessageType.ERROR, {"error": str(exc)})))
            continue

        try:
            if message.type == MessageType.HELLO:
                try:
                    assert_compatible_protocol_version(message.payload, peer="client")
                except RuntimeError as exc:
                    await ws.send(
                        pack_message(
                            Message(
                                MessageType.ERROR,
                                {"error": str(exc)},
                                seq=message.seq,
                            )
                        )
                    )
                    await ws.close(code=1002, reason="protocol version mismatch")
                    return
                await ws.send(
                    pack_message(
                        Message(
                            MessageType.HELLO,
                            hello_payload(
                                role="server",
                                ready=True,
                                action_spec=_serializable_spec(adapter.get_action_spec()),
                                observation_spec=_serializable_spec(adapter.get_observation_spec()),
                            ),
                            seq=message.seq,
                        )
                    )
                )
            elif message.type == MessageType.PING:
                await ws.send(
                    pack_message(
                        Message(
                            MessageType.PONG,
                            {
                                "ready": True,
                                "protocol_version": PROTOCOL_VERSION,
                                "echo": message.payload.get("echo") if isinstance(message.payload, Mapping) else None,
                            },
                            seq=message.seq,
                        )
                    )
                )
            elif message.type == MessageType.EPISODE_START:
                hook = getattr(adapter, "start_episode", None)
                if callable(hook):
                    await _maybe_call(hook, message.payload)
            elif message.type == MessageType.EPISODE_END:
                hook = getattr(adapter, "end_episode", None)
                if callable(hook):
                    await _maybe_call(hook, message.payload)
            elif message.type == MessageType.OBSERVATION:
                obs = message.payload.get("obs") if isinstance(message.payload.get("obs"), Mapping) else message.payload
                instruction = str(
                    message.payload.get("instruction")
                    or message.payload.get("task_description")
                    or message.payload.get("language_instruction")
                    or ""
                )
                action = await asyncio.to_thread(adapter.predict, obs, instruction)
                await ws.send(
                    pack_message(
                        Message(
                            MessageType.ACTION,
                            normalize_action_payload(action),
                            seq=message.seq,
                        )
                    )
                )
            else:
                await ws.send(
                    pack_message(
                        Message(
                            MessageType.ERROR,
                            {"error": f"unsupported message type: {message.type.value}"},
                            seq=message.seq,
                        )
                    )
                )
        except Exception as exc:
            logger.exception("embodied policy server request failed")
            await ws.send(
                pack_message(
                    Message(
                        MessageType.ERROR,
                        {"error": f"{type(exc).__name__}: {exc}"},
                        seq=message.seq,
                    )
                )
            )


async def serve_async(
    adapter: EmbodiedPolicyAdapter,
    *,
    host: str = DEFAULT_HOST,
    port: int = 8000,
    auth_token: str | None = None,
) -> None:
    """Serve a policy adapter over WebSocket until cancelled."""
    import websockets

    required_token = auth_token if auth_token is not None else require_embodied_auth_token_for_host(host)

    async def _process_request(connection: Any, request: Any) -> Any:
        if not required_token:
            return None
        headers = getattr(request, "headers", None)
        candidate = _token_from_headers(headers)
        if token_matches(candidate, required_token):
            return None
        respond = getattr(connection, "respond", None)
        if callable(respond):
            return respond(401, "Unauthorized\n")
        return (401, [("Content-Type", "text/plain")], b"Unauthorized\n")

    def _legacy_process_request(path: str, request_headers: Any) -> tuple[int, list[tuple[str, str]], bytes] | None:
        del path
        if not required_token:
            return None
        candidate = _token_from_headers(request_headers)
        if token_matches(candidate, required_token):
            return None
        return (401, [("Content-Type", "text/plain")], b"Unauthorized\n")

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, adapter)

    serve_kwargs: dict[str, Any] = {
        "compression": None,
        "max_size": DEFAULT_MAX_MESSAGE_BYTES if DEFAULT_MAX_MESSAGE_BYTES > 0 else None,
        "ping_interval": DEFAULT_PING_INTERVAL_S if DEFAULT_PING_INTERVAL_S > 0 else None,
    }
    try:
        async with websockets.serve(
            handler,
            host,
            int(port),
            process_request=_process_request,
            **serve_kwargs,
        ):
            logger.info("Serving embodied policy adapter on ws://%s:%s", host, port)
            await asyncio.Future()
    except TypeError:
        async with websockets.serve(
            handler,
            host,
            int(port),
            process_request=_legacy_process_request,
            **serve_kwargs,
        ):
            logger.info("Serving embodied policy adapter on ws://%s:%s", host, port)
            await asyncio.Future()


def serve(
    adapter: EmbodiedPolicyAdapter,
    *,
    host: str = DEFAULT_HOST,
    port: int = 8000,
    auth_token: str | None = None,
) -> None:
    """Blocking server entry point."""
    asyncio.run(serve_async(adapter, host=host, port=port, auth_token=auth_token))


def load_model_server_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"model server config must be a mapping: {path}")
    return payload


def serve_from_config(path: str | Path, *, host: str | None = None, port: int | None = None) -> None:
    """Build and serve a policy adapter from a YAML model-server config."""
    config = load_model_server_config(path)
    args = dict(config.get("args") or config.get("model_parameters") or {})
    model_id = str(config.get("model_id") or args.pop("model_id", None) or "openvla")
    if host is None:
        host = str(args.pop("host", config.get("host", DEFAULT_HOST)))
    if port is None:
        port = int(args.pop("port", config.get("port", 8000)))
    auth_token = require_embodied_auth_token_for_host(host)
    adapter = build_policy_adapter(model_id, args)
    try:
        serve(adapter, host=host, port=port, auth_token=auth_token)
    finally:
        adapter.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a WorldFoundry embodied policy adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    serve_from_config(args.config, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EMBODIED_SERVER_TOKEN_ENV",
    "DEFAULT_HOST",
    "load_model_server_config",
    "require_embodied_auth_token_for_host",
    "serve",
    "serve_async",
    "serve_from_config",
]
