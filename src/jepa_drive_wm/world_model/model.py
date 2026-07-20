"""
Ego-conditioned VJEPA2.1 world model.

Mimics the VJEPA2-AC world model, adapted for navigation:

- Deterministic prediction of the next spatial latent from context latents
  and ego motion (dx, dy, yaw); further prediction is autoregressive.
- Cross attention: next-frame query tokens attend to context latents
  [Back to the Features], instead of VJEPA2-AC's interleaved self-attention.
- Ego motion conditions every block via AdaLN (DiT adaLN-Zero) [RAE-NWM],
  instead of interleaved action tokens.
- Actions are encoded with deterministic Fourier features [RAE-NWM],
  instead of a linear projection.
- 3D axial RoPE (t, h, w) provides all positional information. Context
  tokens sit at t in [0, T-1]; query tokens sit at t = T, so relative
  spatio-temporal offsets fall out of the q.k dot product.

Deliberate choices:
- Queries are initialised from the last context frame, and the model
  predicts a *residual* on top of it: at initialisation (zero-init output
  projection) the prediction is exactly "copy the last frame" - the right
  prior for small ego motion between frames.
- adaLN-Zero gates every residual branch, so each block is the identity at
  initialisation and the action conditioning fades in during training.

Components:
    FourierEmbedding       - deterministic Fourier features for one scalar
    ActionEmbedder         - (dx, dy, yaw) -> conditioning vector
    AxialRoPE3D            - (t, h, w) rotary embedding for q/k
    CrossAttentionBlockAC  - self-attn + cross-attn + MLP, AdaLN-conditioned
    VJEPA21WorldModel      - main class, with autoregressive rollout
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """Deterministic (non-Gaussian) Fourier embedding for a scalar input.

    Fixed log-spaced frequency bands [2^0, ..., 2^(num_bands-1)] * pi.
    Expects inputs roughly in [-0.5, 0.5] or [0, 1] (normalise first).
    """

    def __init__(self, hidden_size: int, num_bands: int = 16):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_bands)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)  # (num_bands,)
        self.mlp = nn.Sequential(
            nn.Linear(num_bands * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: Float[Tensor, "batch"]) -> Float[Tensor, "batch hidden"]:
        proj: Float[Tensor, "batch bands"] = t.unsqueeze(-1) * self.freqs
        emb: Float[Tensor, "batch 2*bands"] = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return self.mlp(emb)


class ActionEmbedder(nn.Module):
    """Ego motion (dx, dy, yaw) -> conditioning vector, via Fourier features."""

    def __init__(self, hidden_size: int, num_bands: int = 16):
        super().__init__()
        hsize = hidden_size // 3
        self.x_emb = FourierEmbedding(hsize, num_bands)
        self.y_emb = FourierEmbedding(hsize, num_bands)
        self.angle_emb = FourierEmbedding(hidden_size - 2 * hsize, num_bands)

    def forward(self, xya: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch hidden"]:
        x = self.x_emb(xya[..., 0])
        y = self.y_emb(xya[..., 1])
        a = self.angle_emb(xya[..., 2])
        return torch.cat([x, y, a], dim=-1)


# ---------------------------------------------------------------------------
# 3D axial RoPE
# ---------------------------------------------------------------------------

def rotate(
    x: Float[Tensor, "batch heads seq rot_dim"],
    pos: Float[Tensor, "seq"],
) -> Float[Tensor, "batch heads seq rot_dim"]:
    """Rotate the last dim of x by positions `pos` (standard RoPE)."""
    D = x.shape[-1]
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device) / (D / 2.0)
    omega = 1.0 / 10000**omega                                   # (D/2,)
    freq: Float[Tensor, "seq halfdim"] = torch.einsum("n, f -> n f", pos, omega)
    sin = freq.sin().repeat_interleave(2, dim=-1)                # (seq, D)
    cos = freq.cos().repeat_interleave(2, dim=-1)
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + x_rot * sin


class AxialRoPE3D(nn.Module):
    """Axial RoPE over (t, h, w): head_dim is split into three equal chunks,
    each rotated by its own axis position (as in VJEPA2's ACRoPEAttention).
    Any remainder is left unrotated.
    """

    def __init__(self, head_dim: int, grid_size: int = 16):
        super().__init__()
        self.head_dim = head_dim
        self.chunk = int(2 * ((head_dim // 3) // 2))  # even split per axis
        self.grid_size = grid_size

    def make_positions(
        self, T: int, H: int, W: int, device: torch.device, t_offset: float = 0.0
    ) -> tuple[Float[Tensor, "T*H*W"], Float[Tensor, "T*H*W"], Float[Tensor, "T*H*W"]]:
        """(t, h, w) positions for a [T, H, W] grid flattened in that order.
        Spatial positions are snapped to `grid_size` (VJEPA2 convention) so
        RoPE angles are resolution-independent.
        """
        ids = torch.arange(T * H * W, device=device)
        t_pos = (ids // (H * W)).float() + t_offset
        h_pos = ((ids % (H * W)) // W).float() * (self.grid_size / H)
        w_pos = (ids % W).float() * (self.grid_size / W)
        return t_pos, h_pos, w_pos

    def forward(
        self,
        x: Float[Tensor, "batch heads seq head_dim"],
        positions: tuple[Float[Tensor, "seq"], ...],
    ) -> Float[Tensor, "batch heads seq head_dim"]:
        c = self.chunk
        out = [rotate(x[..., i * c:(i + 1) * c], pos) for i, pos in enumerate(positions)]
        if 3 * c < self.head_dim:
            out.append(x[..., 3 * c:])
        return torch.cat(out, dim=-1)


# ---------------------------------------------------------------------------
# Attention + MLP
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Self-attention over the query tokens (one frame) with 3D RoPE."""

    def __init__(self, dim: int, num_heads: int, grid_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rope = AxialRoPE3D(self.head_dim, grid_size)

    def forward(
        self,
        x: Float[Tensor, "batch hw dim"],
        positions: tuple[Float[Tensor, "hw"], ...],
    ) -> Float[Tensor, "batch hw dim"]:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # each [batch heads hw head_dim]
        q, k = self.rope(q, positions), self.rope(k, positions)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    """Cross-attention: queries and keys each rotated by their own 3D positions."""

    def __init__(self, dim: int, num_heads: int, grid_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rope = AxialRoPE3D(self.head_dim, grid_size)

    def forward(
        self,
        x: Float[Tensor, "batch hw dim"],
        context: Float[Tensor, "batch thw dim"],
        q_positions: tuple[Float[Tensor, "hw"], ...],
        kv_positions: tuple[Float[Tensor, "thw"], ...],
    ) -> Float[Tensor, "batch hw dim"]:
        B, Nq, C = x.shape
        Nk = context.shape[1]
        q = self.q(x).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).view(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                        # each [batch heads thw head_dim]
        q, k = self.rope(q, q_positions), self.rope(k, kv_positions)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, Nq, C))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (VJEPA2 convention, incl. 2/3 width + align to 8)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2 * dim * mlp_ratio / 3)
        hidden = (hidden + 7) // 8 * 8
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(dim, hidden)
        self.fc3 = nn.Linear(hidden, dim)

    def forward(self, x: Float[Tensor, "batch hw dim"]) -> Float[Tensor, "batch hw dim"]:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Action-conditioned cross-attention block
# ---------------------------------------------------------------------------

def modulate(
    x: Float[Tensor, "batch hw dim"],
    shift: Float[Tensor, "batch dim"],
    scale: Float[Tensor, "batch dim"],
) -> Float[Tensor, "batch hw dim"]:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttentionBlockAC(nn.Module):
    """Cross-attention block with 3D axial RoPE and AdaLN action conditioning.

    Per block, the query stream (next-frame tokens) runs:
        self-attention over queries   (queries communicate spatially)
        cross-attention onto context  (queries read the past)
        SwiGLU MLP
    Each branch is AdaLN-modulated and gated (adaLN-Zero): the ego-motion
    embedding produces (shift, scale, gate) per branch through a
    zero-initialised linear, so the block is the identity at initialisation.
    """

    def __init__(self, dim: int, num_heads: int, grid_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(dim, num_heads, grid_size)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(dim, num_heads, grid_size)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLUFFN(dim, mlp_ratio)

        # 3 branches x (shift, scale, gate)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))

    def forward(
        self,
        x: Float[Tensor, "batch hw dim"],
        context: Float[Tensor, "batch thw dim"],
        cond: Float[Tensor, "batch dim"],
        q_positions: tuple[Float[Tensor, "hw"], ...],
        kv_positions: tuple[Float[Tensor, "thw"], ...],
    ) -> Float[Tensor, "batch hw dim"]:
        (sa_shift, sa_scale, sa_gate,
         ca_shift, ca_scale, ca_gate,
         mlp_shift, mlp_scale, mlp_gate) = self.adaLN(cond).chunk(9, dim=-1)

        x = x + sa_gate.unsqueeze(1) * self.self_attn(
            modulate(self.norm_sa(x), sa_shift, sa_scale), q_positions
        )
        x = x + ca_gate.unsqueeze(1) * self.cross_attn(
            modulate(self.norm_ca(x), ca_shift, ca_scale), context, q_positions, kv_positions
        )
        x = x + mlp_gate.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(x), mlp_shift, mlp_scale)
        )
        return x


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class VJEPA21WorldModel(nn.Module):
    """Predicts Z_hat_{T+1} from context latents [Z_1..Z_T] and ego motion.

    Queries are initialised from the last context frame and the model predicts
    a residual on top of it, so at initialisation the prediction is exactly
    "copy Z_T". Autoregressive rollout feeds predictions back as context.
    """

    def __init__(
        self,
        grid_height: int = 24,     # KITTI, ViT-B/16 @ 384x1248
        grid_width: int = 78,
        latent_dim: int = 768,     # VJEPA2.1 ViT-B embedding dim
        action_dim: int = 3,       # ego motion (dx, dy, yaw)
        predictor_dim: int = 384,
        depth: int = 6,
        num_heads: int = 8,        # head_dim 48 -> clean 16/16/16 RoPE split
        grid_size: int = 16,       # RoPE spatial normalisation (VJEPA2 convention)
        init_std: float = 0.02,
    ):
        super().__init__()
        assert action_dim == 3, "ActionEmbedder encodes (dx, dy, yaw)"
        self.grid_height = grid_height
        self.grid_width = grid_width

        self.predictor_embed = nn.Linear(latent_dim, predictor_dim, bias=True)
        self.action_embedder = ActionEmbedder(predictor_dim)
        self.blocks = nn.ModuleList(
            CrossAttentionBlockAC(predictor_dim, num_heads, grid_size) for _ in range(depth)
        )

        # Final layer (DiT-style): modulated norm + projection back to latent_dim.
        self.norm_out = nn.LayerNorm(predictor_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(predictor_dim, 2 * predictor_dim))
        self.proj_out = nn.Linear(predictor_dim, latent_dim, bias=True)

        self._init_weights(init_std)

    def _init_weights(self, std: float):
        def base_init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(base_init)
        # adaLN-Zero: conditioning and output start at zero.
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        context_latents: Float[Tensor, "batch time height width latent"],
        ego_motion: Float[Tensor, "batch 3"],
    ) -> Float[Tensor, "batch height width latent"]:
        """One-step prediction: Z_hat_{T+1} = Z_T + f(context, action)."""
        B, T, H, W, C = context_latents.shape
        device = context_latents.device

        # Map tokens to predictor dimension
        ctx: Float[Tensor, "batch thw pred"] = self.predictor_embed(
            context_latents.reshape(B, T * H * W, C)
        )
        cond: Float[Tensor, "batch pred"] = self.action_embedder(ego_motion)

        # Queries: last context frame; RoPE places them at t = T.
        x: Float[Tensor, "batch hw pred"] = ctx.view(B, T, H * W, -1)[:, -1]
        rope = self.blocks[0].cross_attn.rope
        kv_positions = rope.make_positions(T, H, W, device)             # t in [0, T-1]
        q_positions = rope.make_positions(1, H, W, device, t_offset=T)  # t = T

        # Fwd prop
        for blk in self.blocks:
            x = blk(x, ctx, cond, q_positions, kv_positions)

        # Project back to latent dimension; predict residual on top of Z_T.
        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        delta: Float[Tensor, "batch hw latent"] = self.proj_out(
            modulate(self.norm_out(x), shift, scale)
        )
        return context_latents[:, -1] + delta.view(B, H, W, C)

    @torch.no_grad()
    def rollout(
        self,
        context_latents: Float[Tensor, "batch time height width latent"],
        ego_motions: Float[Tensor, "batch steps 3"],
    ) -> Float[Tensor, "batch steps height width latent"]:
        """Autoregressive rollout: each prediction is appended to a sliding
        context window of the original length T. (Call the plain forward
        inside your training loop for the rollout loss - this helper is
        no-grad, for evaluation.)
        """
        T = context_latents.shape[1]
        preds = []
        for k in range(ego_motions.shape[1]):
            z_next = self.forward(context_latents, ego_motions[:, k])
            preds.append(z_next)
            context_latents = torch.cat(
                [context_latents, z_next.unsqueeze(1)], dim=1
            )[:, -T:]
        return torch.stack(preds, dim=1)