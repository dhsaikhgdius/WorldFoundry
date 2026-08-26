"""Reward registry, evaluator, and FastAPI service."""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from worldfoundry.studio.serving.auth import is_loopback_host, request_token_valid

from ..contracts import RewardRequest, RewardResult
from .codec import decode_wire_value, encode_wire_value

REWARD_SERVICE_TOKEN_ENV = "WF_REWARD_SERVICE_TOKEN"
DEFAULT_HOST = "127.0.0.1"


def configured_reward_service_token() -> str:
    """Return the operator-configured shared token ("" when unset)."""

    return os.getenv(REWARD_SERVICE_TOKEN_ENV, "").strip()


def require_reward_auth_token_for_host(host: str) -> str:
    """Fail closed when binding a non-loopback host without a shared token.

    Loopback binds keep the historical unauthenticated behavior and return "".
    Non-loopback binds require ``WF_REWARD_SERVICE_TOKEN``; when it is missing
    this raises ``SystemExit`` so the unauthenticated ``/score`` endpoint is
    never exposed beyond the local host.
    """

    if is_loopback_host(host):
        return ""
    token = configured_reward_service_token()
    if not token:
        raise SystemExit(
            f"reward service refuses to bind non-loopback host {host!r} without authentication. "
            f"Set {REWARD_SERVICE_TOKEN_ENV} to a shared secret (trainers must send "
            "'Authorization: Bearer <token>' on every request), "
            f"or bind host {DEFAULT_HOST}."
        )
    return token


@dataclass(frozen=True, slots=True)
class RewardComponentOutput:
    value: float
    valid: bool = True
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@runtime_checkable
class RewardComponentScorer(Protocol):
    def score(self, requests: tuple[RewardRequest, ...]) -> Sequence[float | RewardComponentOutput]: ...


class RewardScorerRegistry:
    """Named scorer instances shared by local and HTTP execution."""

    def __init__(self) -> None:
        self._scorers: dict[str, RewardComponentScorer | Callable[..., object]] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._scorers))

    def register(
        self,
        name: str,
        scorer: RewardComponentScorer | Callable[..., object],
    ) -> None:
        resolved = str(name).strip()
        if not resolved or resolved in self._scorers:
            raise ValueError(f"reward scorer name is empty or already registered: {name!r}")
        if not callable(scorer) and not callable(getattr(scorer, "score", None)):
            raise TypeError("reward scorer must be callable or expose score(requests)")
        self._scorers[resolved] = scorer

    def scorer(self, name: str) -> RewardComponentScorer | Callable[..., object]:
        try:
            return self._scorers[name]
        except KeyError as error:
            raise KeyError(f"unknown reward scorer {name!r}") from error


def _component_outputs(
    scorer: RewardComponentScorer | Callable[..., object],
    requests: tuple[RewardRequest, ...],
) -> tuple[RewardComponentOutput, ...]:
    if callable(getattr(scorer, "score", None)):
        raw = scorer.score(requests)  # type: ignore[union-attr]
    else:
        raw = scorer(requests)  # type: ignore[operator]
    if inspect.isawaitable(raw):
        raise TypeError("async scorers must be adapted before registration")
    values = tuple(raw)  # type: ignore[arg-type]
    if len(values) != len(requests):
        raise ValueError("reward scorer output count differs from its request count")
    return tuple(
        value if isinstance(value, RewardComponentOutput) else RewardComponentOutput(float(value)) for value in values
    )


class NativeRewardService:
    """Batch by component, run each scorer once, then restore request order."""

    def __init__(self, registry: RewardScorerRegistry, *, fail_fast: bool = True) -> None:
        self.registry = registry
        self.fail_fast = bool(fail_fast)

    def evaluate(self, requests: tuple[RewardRequest, ...]) -> tuple[RewardResult, ...]:
        if not requests:
            return ()
        started = time.perf_counter()
        values = [dict() for _ in requests]
        valid = [dict() for _ in requests]
        diagnostics = [dict() for _ in requests]
        by_reward: dict[str, list[int]] = {}
        for index, request in enumerate(requests):
            for reward_id in request.reward_ids:
                by_reward.setdefault(reward_id, []).append(index)

        for reward_id, indices in by_reward.items():
            selected = tuple(requests[index] for index in indices)
            try:
                outputs = _component_outputs(self.registry.scorer(reward_id), selected)
            except Exception as error:
                if self.fail_fast:
                    raise
                outputs = tuple(
                    RewardComponentOutput(0.0, valid=False, diagnostics={"error": str(error)}) for _ in selected
                )
            for index, output in zip(indices, outputs):
                values[index][reward_id] = float(output.value)
                valid[index][reward_id] = bool(output.valid)
                if output.diagnostics:
                    diagnostics[index][reward_id] = dict(output.diagnostics)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return tuple(
            RewardResult(
                request_id=request.request_id,
                rollout_id=request.rollout_id,
                values=values[index],
                valid=valid[index],
                diagnostics=diagnostics[index],
                latency_ms=latency_ms,
            )
            for index, request in enumerate(requests)
        )


class WorkerGroupRewardScorer:
    """Dispatch component batches across a Ray worker group in round-robin order."""

    def __init__(
        self,
        worker_group: object,
        *,
        method: str = "score",
        batch_size: int = 8,
    ) -> None:
        if not callable(getattr(worker_group, "submit", None)) or not callable(getattr(worker_group, "gather", None)):
            raise TypeError("worker_group must expose submit and gather")
        self.worker_group = worker_group
        self.method = str(method).strip()
        if not self.method:
            raise ValueError("worker scorer method must be non-empty")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("worker scorer batch_size must be positive")

    def score(self, requests: tuple[RewardRequest, ...]) -> tuple[float | RewardComponentOutput, ...]:
        map_batches = getattr(self.worker_group, "map_batches", None)
        if callable(map_batches):
            return tuple(
                map_batches(
                    self.method,
                    requests,
                    batch_size=self.batch_size,
                )
            )
        refs = [self.worker_group.submit(self.method, request) for request in requests]
        return tuple(self.worker_group.gather(refs))


def _request_from_wire(item: object) -> RewardRequest:
    if not isinstance(item, dict):
        raise TypeError("reward wire request must be an object")
    return RewardRequest(
        request_id=str(item["request_id"]),
        rollout_id=str(item["rollout_id"]),
        prompt=str(item["prompt"]),
        conditions=decode_wire_value(item.get("conditions", {})),
        artifacts=decode_wire_value(item.get("artifacts", {})),
        reward_ids=tuple(str(value) for value in item["reward_ids"]),
        metadata=decode_wire_value(item.get("metadata", {})),
    )


def _result_to_wire(result: RewardResult) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "rollout_id": result.rollout_id,
        "values": dict(result.values),
        "valid": dict(result.valid),
        "diagnostics": encode_wire_value(result.diagnostics),
        "latency_ms": result.latency_ms,
    }


def create_reward_service_app(service: NativeRewardService, *, auth_token: str = "") -> object:
    """Build a FastAPI application without importing web dependencies at package import.

    When ``auth_token`` is non-empty every route requires
    ``Authorization: Bearer <token>`` (or ``?token=<token>``) and rejects other
    requests with 401 before any handler runs.
    """

    if not isinstance(service, NativeRewardService):
        raise TypeError("service must be NativeRewardService")
    from fastapi import FastAPI

    app = FastAPI(title="WorldFoundry Reward Service")
    required_token = str(auth_token or "").strip()

    if required_token:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def _require_bearer_token(request: Request, call_next):  # type: ignore[no-untyped-def]
            if request_token_valid(
                required_token,
                authorization_header=request.headers.get("authorization"),
                query_token=request.query_params.get("token"),
            ):
                return await call_next(request)
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "reward_ids": list(service.registry.names)}

    @app.get("/rewards")
    async def rewards() -> dict[str, object]:
        return {"reward_ids": list(service.registry.names)}

    @app.post("/score")
    async def score(payload: dict[str, object]) -> dict[str, object]:
        raw_requests = payload.get("requests")
        if not isinstance(raw_requests, list):
            raise ValueError("score payload requires a requests list")
        requests = tuple(_request_from_wire(item) for item in raw_requests)
        results = await asyncio.to_thread(service.evaluate, requests)
        return {"results": [_result_to_wire(result) for result in results]}

    return app


def serve_reward_service(
    service: NativeRewardService,
    *,
    host: str = DEFAULT_HOST,
    port: int = 8080,
    auth_token: str | None = None,
) -> None:
    """Run the HTTP gateway; model-level parallelism belongs inside scorers.

    The default binding is loopback-only and unauthenticated.  Binding a
    non-loopback host (for trainers on other nodes) additionally requires
    ``WF_REWARD_SERVICE_TOKEN``: the process refuses to start without it, and
    every request must then carry ``Authorization: Bearer <token>``.  Pass
    ``auth_token`` explicitly to override the environment lookup ("" disables
    auth for loopback-style deployments behind an external guard).
    """

    required_token = auth_token if auth_token is not None else require_reward_auth_token_for_host(host)

    import uvicorn

    uvicorn.run(create_reward_service_app(service, auth_token=required_token), host=host, port=port)


__all__ = [
    "REWARD_SERVICE_TOKEN_ENV",
    "NativeRewardService",
    "RewardComponentOutput",
    "RewardComponentScorer",
    "RewardScorerRegistry",
    "WorkerGroupRewardScorer",
    "configured_reward_service_token",
    "create_reward_service_app",
    "require_reward_auth_token_for_host",
    "serve_reward_service",
]
