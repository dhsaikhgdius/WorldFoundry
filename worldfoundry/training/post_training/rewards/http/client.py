"""Synchronous HTTP client implementing the native reward evaluator contract."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import requests

from ..contracts import RewardRequest, RewardResult
from .codec import decode_wire_value, encode_wire_value
from .service import REWARD_SERVICE_TOKEN_ENV


def _request_payload(request: RewardRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "rollout_id": request.rollout_id,
        "prompt": request.prompt,
        "conditions": encode_wire_value(request.conditions),
        "artifacts": encode_wire_value(request.artifacts),
        "reward_ids": list(request.reward_ids),
        "metadata": encode_wire_value(request.metadata),
    }


class HTTPRewardEvaluator:
    """Batch requests to a remote WorldFoundry reward service."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        retry_attempts: int = 0,
        retry_backoff_seconds: float = 0.5,
        trust_environment: bool = False,
        session: requests.Session | None = None,
        auth_token: str | None = None,
    ) -> None:
        url = str(base_url).strip().rstrip("/")
        if not url:
            raise ValueError("base_url must be non-empty")
        if timeout_seconds <= 0 or retry_attempts < 0 or retry_backoff_seconds < 0:
            raise ValueError("HTTP reward timing options are invalid")
        self.base_url = url
        self.timeout_seconds = float(timeout_seconds)
        self.retry_attempts = int(retry_attempts)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        if session is None:
            session = requests.Session()
            session.trust_env = bool(trust_environment)
        self.session = session
        # Remote services bound to non-loopback hosts enforce a shared token;
        # default to the same env var the server reads so call sites need no
        # plumbing. Kept per-request instead of mutating a caller-owned session.
        if auth_token is None:
            auth_token = os.getenv(REWARD_SERVICE_TOKEN_ENV, "")
        token = str(auth_token).strip()
        self._request_headers: dict[str, str] = {"Authorization": f"Bearer {token}"} if token else {}

    def health(self) -> Mapping[str, object]:
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout_seconds,
            headers=self._request_headers or None,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("reward service health response must be an object")
        return payload

    @staticmethod
    def _is_retryable(error: requests.RequestException) -> bool:
        """Retry transient transport failures, not malformed-request rejections."""

        if not isinstance(error, requests.HTTPError) or error.response is None:
            return True
        status = int(error.response.status_code)
        return status >= 500 or status in {408, 429}

    def evaluate(self, requests_: tuple[RewardRequest, ...]) -> tuple[RewardResult, ...]:
        if not requests_:
            return ()
        payload = {"requests": [_request_payload(request) for request in requests_]}
        response: Any = None
        for attempt in range(self.retry_attempts + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/score",
                    json=payload,
                    timeout=self.timeout_seconds,
                    headers=self._request_headers or None,
                )
                response.raise_for_status()
                break
            except requests.RequestException as error:
                if attempt == self.retry_attempts or not self._is_retryable(error):
                    raise
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        body = response.json()
        raw_results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(raw_results, list) or len(raw_results) != len(requests_):
            raise ValueError("reward service returned a result count different from the request batch")
        return tuple(
            RewardResult(
                request_id=str(item["request_id"]),
                rollout_id=str(item["rollout_id"]),
                values=dict(item["values"]),
                valid=dict(item["valid"]),
                diagnostics=decode_wire_value(item.get("diagnostics", {})),
                latency_ms=float(item["latency_ms"]),
            )
            for item in raw_results
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> HTTPRewardEvaluator:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()


__all__ = ["HTTPRewardEvaluator"]
