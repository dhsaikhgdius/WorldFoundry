"""Async WebSocket client for embodied policy servers."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .protocol import (
    Message,
    MessageType,
    assert_compatible_protocol_version,
    hello_payload,
    pack_message,
    unpack_message,
)

logger = logging.getLogger(__name__)

_DEFAULT_PING_INTERVAL_S = float(os.getenv("WF_EMBODIED_WS_PING_INTERVAL_S", "20") or "20")
_DEFAULT_MAX_MESSAGE_BYTES = int(os.getenv("WF_EMBODIED_WS_MAX_SIZE", str(64 * 1024 * 1024)))


class EmbodiedWebSocketConnection:
    """Small WebSocket client for the WorldFoundry embodied episode protocol."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        benchmark: str | None = None,
        ping_interval: float | None = None,
        max_size: int | None = None,
    ) -> None:
        self.url = str(url)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.benchmark = benchmark
        self.ping_interval = _DEFAULT_PING_INTERVAL_S if ping_interval is None else float(ping_interval)
        self.max_size = _DEFAULT_MAX_MESSAGE_BYTES if max_size is None else int(max_size)
        self.server_info: dict[str, Any] = {}
        self._seq = 0
        self._ws: Any = None

    async def connect(self) -> None:
        await self._connect_with_backoff()
        await self._hello()

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def reconnect(self) -> None:
        await self.close()
        await self.connect()

    async def send(self, message_type: MessageType, payload: dict[str, Any], *, seq: int | None = None) -> int:
        if self._ws is None:
            await self.connect()
        if seq is None:
            self._seq += 1
            seq = self._seq
        message = Message(type=message_type, payload=payload, seq=seq)
        await self._ws.send(pack_message(message))
        return seq

    async def recv(self, *, timeout: float | None = None) -> Message:
        if self._ws is None:
            raise RuntimeError("WebSocket connection is not open")
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout if timeout is None else timeout)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return unpack_message(raw)

    async def start_episode(self, config: dict[str, Any]) -> None:
        await self.send(MessageType.EPISODE_START, config)

    async def end_episode(self, result: dict[str, Any]) -> None:
        await self.send(MessageType.EPISODE_END, result)

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        seq = await self.send(MessageType.OBSERVATION, obs)
        response = await self.recv(timeout=self.timeout)
        if response.type == MessageType.ERROR:
            raise RuntimeError(f"policy server error: {response.payload}")
        if response.type != MessageType.ACTION:
            raise RuntimeError(f"expected action response, got {response.type.value}")
        if response.seq != seq:
            logger.warning("policy server seq mismatch: sent %s got %s", seq, response.seq)
        return response.payload

    async def ping(self, *, echo: Any = None) -> dict[str, Any]:
        """Application-level readiness probe (PING → PONG)."""

        payload: dict[str, Any] = {}
        if echo is not None:
            payload["echo"] = echo
        seq = await self.send(MessageType.PING, payload)
        reply = await self.recv(timeout=self.timeout)
        if reply.type == MessageType.ERROR:
            raise RuntimeError(f"policy server error: {reply.payload}")
        if reply.type != MessageType.PONG:
            raise RuntimeError(f"expected PONG reply, got {reply.type.value}")
        if reply.seq != seq:
            logger.warning("policy server ping seq mismatch: sent %s got %s", seq, reply.seq)
        return dict(reply.payload or {})

    async def _hello(self) -> None:
        payload = hello_payload(role="client", **({"benchmark": self.benchmark} if self.benchmark else {}))
        await self.send(MessageType.HELLO, payload)
        reply = await self.recv(timeout=self.timeout)
        if reply.type == MessageType.ERROR:
            raise RuntimeError(f"policy server HELLO error: {reply.payload}")
        if reply.type != MessageType.HELLO:
            raise RuntimeError(f"expected HELLO reply, got {reply.type.value}")
        assert_compatible_protocol_version(reply.payload, peer="server")
        self.server_info = dict(reply.payload or {})

    async def _connect_with_backoff(self) -> None:
        import websockets

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.url,
                        compression=None,
                        max_size=self.max_size if self.max_size > 0 else None,
                        ping_interval=self.ping_interval if self.ping_interval > 0 else None,
                    ),
                    timeout=self.timeout,
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                await asyncio.sleep(self.backoff_base**attempt)
        raise ConnectionError(f"policy server unreachable at {self.url}") from last_exc


__all__ = ["EmbodiedWebSocketConnection"]
