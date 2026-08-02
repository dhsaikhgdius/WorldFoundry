"""Native checkpoint-compatible Wan diffusion transformer."""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.core.attention import (
    apply_complex_rotary_embedding as rope_apply,
)
from worldfoundry.core.attention import (
    complex_rotary_frequencies_3d as precompute_freqs_cis_3d,
)
from worldfoundry.core.attention import (
    packed_sequence_attention as flash_attention,
)
from worldfoundry.core.gradient import gradient_checkpoint_forward
from worldfoundry.core.nn import RMSNorm, sinusoidal_embedding_1d
from worldfoundry.core.nn import scale_shift as modulate

from .adapter import SimpleAdapter


class AttentionModule(nn.Module):
    """Attention module implementation."""

    def __init__(self, num_heads):
        """Init.

        Args:
            num_heads: The num heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.compatibility_mode = False

    def forward(self, q, k, v):
        """Forward.

        Args:
            q: The q.
            k: The k.
            v: The v.
        """
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            compatibility_mode=self.compatibility_mode,
        )
        return x


class SelfAttention(nn.Module):
    """Self attention implementation."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        """Forward.

        Args:
            x: The x.
            freqs: The freqs.
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttentionProcessor:
    """Default Wan cross-attention policy, replaceable by research adapters."""

    def __call__(
        self,
        attention: "CrossAttention",
        x: torch.Tensor,
        context: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if attention.has_image_input:
            image_context = context[:, :257]
            text_context = context[:, 257:]
        else:
            text_context = context
        query = attention.norm_q(attention.q(x))
        key = attention.norm_k(attention.k(text_context))
        value = attention.v(text_context)
        output = attention.attn(query, key, value)
        if attention.has_image_input:
            image_key = attention.norm_k_img(attention.k_img(image_context))
            image_value = attention.v_img(image_context)
            output = output + flash_attention(
                query,
                image_key,
                image_value,
                num_heads=attention.num_heads,
            )
        return attention.o(output)


class CrossAttention(nn.Module):
    """Cross attention implementation."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            eps: The eps.
            has_image_input: The has image input.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.processor = CrossAttentionProcessor()

    def set_processor(self, processor: Any) -> None:
        """Install a model-specific cross-attention policy."""

        self.processor = processor

    def get_processor(self) -> Any:
        return self.processor

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any):
        """Forward.

        Args:
            x: The x.
            y: The y.
        """
        return self.processor(self, x, y, **kwargs)


class GateModule(nn.Module):
    """Gate module implementation."""

    def __init__(
        self,
    ):
        """Init."""
        super().__init__()

    def forward(self, x, gate, residual):
        """Forward.

        Args:
            x: The x.
            gate: The gate.
            residual: The residual.
        """
        return x + gate * residual


class DiTBlock(nn.Module):
    """Di t block implementation."""

    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        """Init.

        Args:
            has_image_input: The has image input.
            dim: The dim.
            num_heads: The num heads.
            ffn_dim: The ffn dim.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context=None,
        t_mod=None,
        freqs=None,
        *,
        return_partial: bool = False,
        run_remaining: bool = False,
        modifiers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        """Forward.

        Args:
            x: The x.
            context: The context.
            t_mod: The t mod.
            freqs: The freqs.
        """
        if run_remaining:
            if modifiers is None:
                raise ValueError("Wan block modifiers are required for run_remaining")
            shift_mlp, scale_mlp, gate_mlp = modifiers
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            return self.gate(x, gate_mlp, self.ffn(input_x))

        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context, **kwargs)
        if return_partial:
            return x, (shift_mlp, scale_mlp, gate_mlp)
        if modifiers is not None:
            shift_mlp, scale_mlp, gate_mlp = modifiers
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

    def forward_partial(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs, return_partial=True)

    def forward_remaining(
        self,
        x: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            x,
            run_remaining=True,
            modifiers=(shift_mlp, scale_mlp, gate_mlp),
        )


class MLP(torch.nn.Module):
    """Mlp implementation."""

    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        """Init.

        Args:
            in_dim: The in dim.
            out_dim: The out dim.
            has_pos_emb: The has pos emb.
        """
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    """Head implementation."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        """Init.

        Args:
            dim: The dim.
            out_dim: The out dim.
            patch_size: The patch size.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        """Forward.

        Args:
            x: The x.
            t_mod: The t mod.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            head_dtype = self.head.weight.dtype
            if len(t_mod.shape) == 3:
                shift, scale = (
                    self.modulation.unsqueeze(0).to(
                        device=t_mod.device,
                        dtype=torch.float32,
                    )
                    + t_mod.float().unsqueeze(2)
                ).chunk(2, dim=2)
                x = self.head(
                    (
                        self.norm(x.float()) * (1 + scale.squeeze(2))
                        + shift.squeeze(2)
                    ).to(dtype=head_dtype)
                )
            else:
                shift, scale = (
                    self.modulation.to(device=t_mod.device, dtype=torch.float32)
                    + t_mod.float()
                ).chunk(2, dim=1)
                x = self.head(
                    (self.norm(x.float()) * (1 + scale) + shift).to(dtype=head_dtype)
                )
        return x


class WanModel(torch.nn.Module):
    """Wan model implementation."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        inject_sample_info: bool = False,
        per_token_timestep: bool = False,
        diffusion_model_pretrained_path: str | None = None,
        block_class: type[nn.Module] | None = None,
        block_kwargs: dict[str, Any] | None = None,
    ):
        """Init.

        Args:
            dim: The dim.
            in_dim: The in dim.
            ffn_dim: The ffn dim.
            out_dim: The out dim.
            text_dim: The text dim.
            freq_dim: The freq dim.
            eps: The eps.
            patch_size: The patch size.
            num_heads: The num heads.
            num_layers: The num layers.
            has_image_input: The has image input.
            has_image_pos_emb: The has image pos emb.
            has_ref_conv: The has ref conv.
            add_control_adapter: The add control adapter.
            in_dim_control_adapter: The in dim control adapter.
            seperated_timestep: The seperated timestep.
            require_vae_embedding: The require vae embedding.
            require_clip_embedding: The require clip embedding.
            fuse_vae_embedding_in_latents: The fuse vae embedding in latents.
        """
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.inject_sample_info = bool(inject_sample_info)
        self.per_token_timestep = bool(per_token_timestep)
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        if self.inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        block_class = block_class or DiTBlock
        block_kwargs = dict(block_kwargs or {})
        self.blocks = nn.ModuleList(
            [
                block_class(
                    has_image_input,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    **block_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:]
            )
        else:
            self.control_adapter = None

    def set_attention_compatibility_mode(self, enabled: bool) -> None:
        """Select exact PyTorch SDPA for this model instance.

        The shared attention dispatcher reads its default backend at import
        time.  Training must therefore bind the correctness path on the
        actual Wan modules instead of relying on a late environment change.
        """

        if not isinstance(enabled, bool):
            raise TypeError("Wan attention compatibility mode must be a bool")
        for module in self.modules():
            if isinstance(module, AttentionModule):
                module.compatibility_mode = enabled

    def enable_control_adapter(self, in_dim: int = 24) -> None:
        """Attach the canonical Wan spatial control role after base loading.

        Some releases publish the adapter only in an overlay checkpoint, so it
        cannot participate in strict restoration of the official base DiT.
        """

        if self.control_adapter is not None:
            return
        reference = self.patch_embedding.weight
        self.control_adapter = SimpleAdapter(
            in_dim,
            self.dim,
            kernel_size=self.patch_size[1:],
            stride=self.patch_size[1:],
        ).to(device=reference.device, dtype=reference.dtype)

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        """Patchify.

        Args:
            x: The x.
            control_camera_latents_input: The control camera latents input.
        """
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        """Unpatchify.

        Args:
            x: The x.
            grid_size: The grid size.
        """
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def block_forward_kwargs(
        self,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return variant-specific keyword arguments for every transformer block.

        Wan research variants can override this hook to add conditioning without
        copying the common patchification, RoPE, timestep, and output path.
        """

        del grid_size, kwargs
        return {}

    def prepare_condition_context(
        self,
        context: torch.Tensor | None,
        *,
        clip_feature: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Project the cross-attention condition used by the shared Wan blocks.

        Most Wan releases use UMT5 (and optionally CLIP) through the canonical
        projection below.  Research variants whose checkpoint already emits
        DiT-width condition tokens can override this hook without copying the
        patchification, timestep, RoPE, transformer, and output path.
        """

        del kwargs
        if context is None:
            raise ValueError("Wan text-conditioned inference requires a context tensor")
        context = self.text_embedding(context)
        if self.has_image_input:
            if clip_feature is None:
                raise ValueError("Wan image-conditioned inference requires clip_feature")
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)
        return context

    def rotary_frequencies(
        self,
        grid_size: tuple[int, int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Build canonical 3D rotary frequencies for one patch grid."""

        f, h, w = grid_size
        return (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(device)
        )

    def prepare_token_sequence(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        t: torch.Tensor,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Variant hook for prefix/suffix conditioning token composition."""

        del kwargs
        return x, freqs, t_mod, t, None

    def finalize_token_sequence(
        self,
        x: torch.Tensor,
        token_state: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Variant hook for extracting auxiliary and target token branches."""

        del token_state, kwargs
        return x, None

    def after_transformer_block(
        self,
        x: torch.Tensor,
        block_id: int,
        token_state: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Variant hook for residual controllers applied between Wan blocks."""

        del block_id, token_state, kwargs
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        memory_context=None,
        fps: Optional[torch.Tensor] = None,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward.

        Args:
            x: The x.
            timestep: The timestep.
            context: The context.
            clip_feature: The clip feature.
            y: The y.
            use_gradient_checkpointing: The use gradient checkpointing.
            use_gradient_checkpointing_offload: The use gradient checkpointing offload.
            memory_context: Optional model-specific memory condition.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Native Wan checkpoints can be loaded directly in BF16 (SCOPE
            # does this to fit the full 14B model).  Hard-casting the
            # sinusoidal features to FP32 while autocast is explicitly
            # disabled makes the first Linear fail with Float/BFloat16.  Use
            # the actual embedding weight dtype so both FP32 and BF16 model
            # loading remain valid.
            time_dtype = next(self.time_embedding.parameters()).dtype
            if timestep.ndim == 1:
                t = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, timestep).to(time_dtype)
                )
                t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
            elif timestep.ndim == 2 and self.per_token_timestep:
                batch, sequence = timestep.shape
                t = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(time_dtype)
                ).unflatten(0, (batch, sequence))
                t_mod = self.time_projection(t).unflatten(2, (6, self.dim))
            else:
                raise ValueError(
                    "Wan timestep must be [B], or [B,L] when per-token timesteps are enabled"
                )
        if self.inject_sample_info:
            if fps is None:
                raise ValueError("Wan sample-info conditioning requires fps ids")
            fps_ids = fps.to(device=x.device, dtype=torch.long).reshape(-1)
            if fps_ids.numel() == 1 and x.shape[0] != 1:
                fps_ids = fps_ids.expand(x.shape[0])
            if fps_ids.numel() != x.shape[0]:
                raise ValueError("Wan fps ids must be scalar or have one value per sample")
            fps_features = self.fps_embedding(fps_ids)
            fps_dtype = next(self.fps_projection.parameters()).dtype
            fps_mod = self.fps_projection(fps_features.to(fps_dtype)).unflatten(1, (6, self.dim))
            t_mod = t_mod + fps_mod
        context = self.prepare_condition_context(
            context,
            clip_feature=clip_feature,
            **kwargs,
        )

        # Wan2.1 I2V uses both CLIP and VAE conditions, while Wan2.2 Fun I2V
        # keeps only the VAE branch.  Treat those roles independently so a
        # native model does not need a second DiT implementation merely to
        # omit CLIP.
        if y is not None and self.require_vae_embedding:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        elif self.has_image_input and self.require_vae_embedding:
            raise ValueError("Wan image-conditioned inference requires VAE condition latents")

        x, (f, h, w) = self.patchify(
            x,
            control_camera_latents_input=control_camera_latents_input,
        )

        memory_adapter = getattr(self, "memory_adapter", None)
        if memory_adapter is not None:
            x, context, memory_context = memory_adapter.prepare_inputs(
                x=x,
                context=context,
                grid_size=(f, h, w),
                memory_context=memory_context,
            )

        freqs = self.rotary_frequencies((f, h, w), device=x.device)
        x, freqs, t_mod, t, token_state = self.prepare_token_sequence(
            x,
            freqs,
            t_mod,
            t,
            (f, h, w),
            context=context,
            **kwargs,
        )

        block_kwargs = self.block_forward_kwargs(
            (f, h, w),
            **kwargs,
        )
        if memory_context is not None:
            block_kwargs["memory_context"] = memory_context

        for block_id, block in enumerate(self.blocks):
            block_inputs = (x, context, t_mod, freqs)
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    *block_inputs,
                    **block_kwargs,
                )
            else:
                x = block(*block_inputs, **block_kwargs)
            x = self.after_transformer_block(
                x,
                block_id,
                token_state,
                **kwargs,
            )

        x, auxiliary = self.finalize_token_sequence(x, token_state, **kwargs)
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return (x, auxiliary) if auxiliary is not None else x

    @staticmethod
    def state_dict_converter():
        from ...denoisers.wan import WanModelStateDictConverter

        return WanModelStateDictConverter()

    @property
    def attn_processors(self) -> dict[str, Any]:
        """Return replaceable cross-attention processors by module path."""

        return {
            f"blocks.{index}.cross_attn.processor": block.cross_attn.get_processor()
            for index, block in enumerate(self.blocks)
            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "get_processor")
        }

    def set_attn_processor(self, processor: Any) -> None:
        """Install one processor everywhere or a path-keyed processor mapping."""

        if isinstance(processor, dict):
            expected = set(self.attn_processors)
            provided = set(processor)
            if provided != expected:
                missing = sorted(expected - provided)
                unexpected = sorted(provided - expected)
                raise ValueError(
                    f"Wan attention processor paths mismatch; missing={missing}, "
                    f"unexpected={unexpected}"
                )
            for index, block in enumerate(self.blocks):
                path = f"blocks.{index}.cross_attn.processor"
                if path in processor:
                    block.cross_attn.set_processor(processor[path])
            return
        for block in self.blocks:
            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "set_processor"):
                block.cross_attn.set_processor(processor)
