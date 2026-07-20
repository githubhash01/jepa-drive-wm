# VJEPA2.1 Ego-Conditioned World Model

Deterministic latent world model for navigation. Given `T` frozen VJEPA2.1
ViT-B latents and an ego motion `(dx, dy, yaw)`, it predicts the next latent;
longer horizons are produced autoregressively. It departs from VJEPA2-AC in
three ways: cross-attention from next-frame queries onto context (instead of
interleaved self-attention), AdaLN action conditioning (instead of action
tokens), and deterministic Fourier action encoding (instead of a linear map).

Shapes below use KITTI ViT-B/16 defaults: `T` = context length, `H = 24`,
`W = 78`, latent dim `C = 768`, predictor dim `D = 384`, depth 6, 8 heads.

## Dataflow

```
context latents  [B, T, 24, 78, 768]        ego motion [B, 3]
        │                                          │
  predictor_embed (Linear 768 -> 384)        ActionEmbedder
        │                                          │
  ctx [B, T·1872, 384]                       cond [B, 384]
        │
  queries  = last frame of ctx  [B, 1872, 384]     (Z_T, at RoPE t = T)
        │
  6 × CrossAttentionBlockAC(queries, ctx, cond)
        │
  final AdaLN(cond) → proj_out (Linear 384 -> 768, zero-init)
        │
  Z_hat_{T+1} = Z_T + delta   [B, 24, 78, 768]
```

## ActionEmbedder — `[B, 3] -> [B, 384]`

Each scalar (`dx`, `dy`, `yaw`) is passed through a deterministic Fourier
embedding: 16 log-spaced bands `[2^0 .. 2^15]·π`, giving sin/cos features
`[B, 32]`, then an MLP (`Linear -> SiLU -> Linear`) to `[B, 128]`. The three
components are concatenated to `[B, 384]`. Inputs are expected normalised to
roughly `[-0.5, 0.5]`.

## 3D Axial RoPE

Head dim 48 is split into 16/16/16 for the `(t, h, w)` axes; each chunk is
rotated by its axis position (spatial positions snapped to `grid_size = 16`,
VJEPA2 convention). Context tokens sit at `t ∈ [0, T-1]`, all query tokens at
`t = T`, so relative spatio-temporal offsets emerge in the q·k dot product —
no learned positional embeddings anywhere.

## CrossAttentionBlockAC (×6)

Per block, the query stream `x [B, 1872, 384]` runs three residual branches:

```
x ← x + g_sa  · SelfAttn( AdaLN(x; cond) )                    queries mix spatially
x ← x + g_ca  · CrossAttn( AdaLN(x; cond), ctx )              queries read the past
x ← x + g_mlp · SwiGLU( AdaLN(x; cond) )
```

`cond` produces `(shift, scale, gate)` for each branch (9 × 384 values) via a
zero-initialised linear (adaLN-Zero): every block is the identity at
initialisation and conditioning fades in during training. Cross-attention
keys/values come from all `T·1872` context tokens; queries and keys are each
RoPE-rotated by their own positions.

## Output & initialisation

The final layer applies one more AdaLN (shift/scale from `cond`), then a
zero-initialised projection back to 768. The result is added to `Z_T`, so the
model predicts a **residual** and the initial prediction is exactly
"copy the last frame" — a strong prior for small inter-frame ego motion.

## Rollout — `[B, T, 24, 78, 768], [B, K, 3] -> [B, K, 24, 78, 768]`

Each prediction is appended to a sliding context window of fixed length `T`.
The `rollout` method is `no_grad` (evaluation); for the rollout loss, call
`forward` iteratively inside the training loop so gradients flow through the
fed-back predictions.

Parameter count at defaults: ~23M.