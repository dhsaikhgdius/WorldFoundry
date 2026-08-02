"""Reward-Forcing bridge to the in-tree Wan EMA-Sink graph."""

from __future__ import annotations

from math import isclose, isfinite

from torch import nn

from ..self_forcing.wan import WanSelfForcingChunkAdapter
from .config import RewardForcingConfig


class WanRewardForcingChunkAdapter(WanSelfForcingChunkAdapter):
    """Reuse native causal cache execution and require released EMA-Sink behavior."""

    def __init__(
        self,
        module: nn.Module,
        config: RewardForcingConfig,
        *,
        checkpoint_identity: str | None = None,
    ) -> None:
        if not isinstance(config, RewardForcingConfig):
            raise TypeError("config must be RewardForcingConfig")
        super().__init__(
            module,
            frames_per_block=config.frames_per_block,
            checkpoint_identity=checkpoint_identity,
        )
        if bool(getattr(self._graph, "independent_first_frame", False)):
            raise ValueError("released Reward-Forcing T2V requires independent_first_frame=False")
        active_block_size = int(getattr(self._graph, "num_frame_per_block", self.frames_per_block))
        if getattr(self._graph, "block_mask", None) is not None and active_block_size != self.frames_per_block:
            raise RuntimeError("cannot change Reward-Forcing Wan block size after caching an attention mask")
        # The official Re-DMD constructor writes this value into the causal
        # graph when the released three-frame blocks are selected.  Keeping it
        # only on the rollout wrapper would make full-graph and KV execution
        # disagree about block boundaries.
        self._graph.num_frame_per_block = self.frames_per_block
        self.audit_reward_forcing_cache(
            frames_per_block=config.frames_per_block,
            local_attention_frames=config.local_attention_frames,
            ema_sink_frames=config.ema_sink_frames,
            ema_sink_decay=config.ema_sink_decay,
        )

    def audit_reward_forcing_cache(
        self,
        *,
        frames_per_block: int,
        local_attention_frames: int,
        ema_sink_frames: int,
        ema_sink_decay: float,
    ) -> None:
        if int(frames_per_block) != self.frames_per_block:
            raise ValueError("Reward-Forcing rollout and Wan block sizes differ")
        if int(getattr(self._graph, "num_frame_per_block", -1)) != int(frames_per_block):
            raise ValueError("Reward-Forcing Wan graph and rollout block sizes differ")
        if bool(getattr(self._graph, "independent_first_frame", False)):
            raise ValueError("released Reward-Forcing T2V requires independent_first_frame=False")
        if int(getattr(self._graph, "local_attn_size", -1)) != int(local_attention_frames):
            raise ValueError("Reward-Forcing Wan local-attention window differs from the active config")
        blocks = tuple(self._graph.blocks)
        if not blocks:
            raise ValueError("Reward-Forcing Wan graph has no transformer blocks")
        sink_values: set[int] = set()
        decay_values: set[float] = set()
        for block in blocks:
            attention = getattr(block, "self_attn", None)
            if attention is None or not callable(getattr(attention, "incremental_update", None)):
                raise TypeError("Reward-Forcing Wan self-attention must execute EMA-Sink updates")
            sink_values.add(int(getattr(attention, "sink_size", -1)))
            decay = float(getattr(attention, "compression_alpha", float("nan")))
            if not isfinite(decay):
                raise ValueError("Reward-Forcing Wan EMA-Sink decay is invalid")
            decay_values.add(decay)
        if sink_values != {int(ema_sink_frames)}:
            raise ValueError("Reward-Forcing Wan EMA-Sink frame count differs from the active config")
        if len(decay_values) != 1 or not isclose(
            next(iter(decay_values)),
            float(ema_sink_decay),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Reward-Forcing Wan EMA-Sink decay differs from the active config")


__all__ = ["WanRewardForcingChunkAdapter"]
