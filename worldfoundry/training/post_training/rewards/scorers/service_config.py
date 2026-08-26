"""Configuration loading for the native reward scorer HTTP service."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .agentic import AgenticCorrectnessConfig, AgenticToolSuccessConfig
from .config import CLAPConfig, VideoPickScoreConfig


@dataclass(frozen=True, slots=True)
class ScorerServiceConfig:
    """Network settings and reward components hosted by one service process."""

    host: str
    port: int
    fail_fast: bool
    videopickscore: VideoPickScoreConfig | None = None
    clap: CLAPConfig | None = None
    correctness: AgenticCorrectnessConfig | None = None
    tool_success: AgenticToolSuccessConfig | None = None

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("reward service host must be non-empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("reward service port must be between 1 and 65535")
        if all(
            scorer is None
            for scorer in (
                self.videopickscore,
                self.clap,
                self.correctness,
                self.tool_success,
            )
        ):
            raise ValueError("reward service requires at least one scorer")

    @property
    def scorer_names(self) -> tuple[str, ...]:
        names = []
        if self.videopickscore is not None:
            names.append("videopickscore")
        if self.clap is not None:
            names.append("clap")
        if self.correctness is not None:
            names.append("correctness")
        if self.tool_success is not None:
            names.append("tool-success")
        return tuple(names)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ScorerServiceConfig:
        unknown_top_level = set(payload) - {"server", "scorers"}
        if unknown_top_level:
            raise ValueError(f"unsupported reward service fields: {sorted(unknown_top_level)}")
        server = payload.get("server", {})
        scorers = payload.get("scorers")
        if not isinstance(server, Mapping):
            raise TypeError("reward service server config must be a mapping")
        if not isinstance(scorers, Mapping):
            raise TypeError("reward service scorers config must be a mapping")
        unknown_server = set(server) - {"host", "port", "fail_fast"}
        if unknown_server:
            raise ValueError(f"unsupported reward server fields: {sorted(unknown_server)}")

        unknown = set(scorers) - {
            "videopickscore",
            "clap",
            "correctness",
            "tool-success",
        }
        if unknown:
            raise ValueError(f"unsupported reward scorers: {sorted(unknown)}")

        video_config = _scorer_config(scorers, "videopickscore", VideoPickScoreConfig)
        clap_config = _scorer_config(scorers, "clap", CLAPConfig)
        correctness_config = _scorer_config(
            scorers,
            "correctness",
            AgenticCorrectnessConfig,
        )
        tool_success_config = _scorer_config(
            scorers,
            "tool-success",
            AgenticToolSuccessConfig,
        )
        return cls(
            # Loopback by default: the /score endpoint only requires a token on
            # non-loopback binds, so an omitted server.host must not expose it.
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8080)),
            fail_fast=bool(server.get("fail_fast", True)),
            videopickscore=video_config,
            clap=clap_config,
            correctness=correctness_config,
            tool_success=tool_success_config,
        )

    def override(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        fail_fast: bool | None = None,
        device: str | None = None,
    ) -> ScorerServiceConfig:
        video_config = self.videopickscore
        clap_config = self.clap
        if device is not None:
            if video_config is not None:
                video_config = replace(video_config, device=device)
            if clap_config is not None:
                clap_config = replace(clap_config, device=device)
        return replace(
            self,
            host=self.host if host is None else host,
            port=self.port if port is None else port,
            fail_fast=self.fail_fast if fail_fast is None else fail_fast,
            videopickscore=video_config,
            clap=clap_config,
        )


def _scorer_config(
    scorers: Mapping[str, object],
    name: str,
    config_type: type[VideoPickScoreConfig | CLAPConfig | AgenticCorrectnessConfig | AgenticToolSuccessConfig],
) -> VideoPickScoreConfig | CLAPConfig | AgenticCorrectnessConfig | AgenticToolSuccessConfig | None:
    payload = scorers.get(name)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError(f"reward scorer {name!r} config must be a mapping")
    return config_type(**dict(payload))


def load_scorer_service_config(path: str | Path) -> ScorerServiceConfig:
    """Load a scorer service YAML or JSON file."""

    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("reward service config must be YAML or JSON")
    if not isinstance(payload, Mapping):
        raise TypeError("reward service config must contain a mapping")
    return ScorerServiceConfig.from_mapping(payload)


__all__ = ["ScorerServiceConfig", "load_scorer_service_config"]
