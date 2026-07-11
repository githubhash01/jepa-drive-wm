# VJEPA feature-space action-conditioned world model — go-forward plan

## Summary

Build a driving world model that predicts the *future* of a scene in V-JEPA 2.1 **feature space**,
conditioned on **ego-motion**, using a **VJEPA2-AC-style** action-conditioned predictor. Frozen
per-task **DPT heads** decode predicted future features into dense maps (depth first;
semantics/normals later). Lineage: DINO-WM / DINO-Foresight, re-instantiated with VJEPA features +
an action-conditioned predictor. Reviewed as feasible and well-grounded.

## Context / why

The only large model (the frozen ViT-B encoder) runs **once** and is cached; everything trained
afterwards (compressor, DPT heads, AC predictor) is small. This is designed to run on a **single
student GPU**, not lab hardware. The dominant unknown — does compressing the feature state lose
task information — is cheap to test and gates everything, so it goes first.

## Locked design decisions

- **State** = per-frame, image-mode (T=1) features from tapped VJEPA layers (start 4: [2,5,8,11];
  de-risk whether 2 suffice). Per-frame, not clip/tubelet, for clean frame↔action↔decode alignment.
- **Full spatial resolution — NO pooling.** Keep the native 24×78 patch grid throughout. Patch
  resolution is preserved end to end; compute is saved on other axes (below), never by coarsening.
- **Compression = channel only, aggressive.** Per-layer PCA (reuse `data/global_pca.py`), target
  well below ¼ — the de-risk sweeps 768→{192,96,64} per layer and 4→2 layers and picks the smallest
  that keeps depth AbsRel within ~5–10% (candidate total ≈128–256-d). PCA is free, per-token
  (spatial grid untouched), and orthonormal (well-conditioned prediction target).
- **Loss = per-patch / spatial, never pooled.** Supervise every token of the predicted latent grid
  against its target (per-token MSE + cosine) over all T×H×W tokens. Pipeline is spatial throughout:
  PCA per token → AC predictor per token → DPT per token → dense output.
- **Predictor = `vit_ac_predictor` (`src/models/ac_predictor.py`), from scratch, tiny.** Depth ~6,
  predictor_embed_dim ~256, ~4–6 heads (vs default depth-24/width-1024). Same-space (in==out),
  block-causal over time, one action + one state token per frame. **Predict the residual** (Δ from
  current frame) for data efficiency. Short sequences **T ≈ 4–8**.
- **Conditioning = ego-motion.** `action` = inter-frame relative pose; `state` = velocity/yaw-rate.
  `use_extrinsics=False` (cond_tokens=2). `action`/`state` share `action_embed_dim` (pad shorter).
- **Cache once.** Encode all frames through frozen ViT-B once → store compressed latents → WM/probe
  training never re-runs the encoder.

## Compute strategy (single-GPU, full-grid)

Full 24×78 grid means ~1,872 tokens/frame; at T=6 a sequence is ~11k tokens. A tiny (depth-6,
width-256) transformer with SDPA/flash + block-causal mask handles this on one GPU. VRAM knobs, in
order: **channel dim** (smaller PCA), **T** (4 vs 8), **batch size + grad-accum**, and
`use_activation_checkpointing`. If full-grid attention ever bottlenecks, a later optimization is
**factorized space/time attention** (spatial-within-frame + temporal-across-frames) — not needed
for v1. Cached latents at ~128-d over the full grid are a few GB on disk.

## Architecture fit (addressing the manipulation concern)

Nothing in the AC predictor is manipulation-specific: the manipulation flavor is only the DROID
weights (train from scratch → gone), `action_embed_dim=7` = ee-pose+gripper (just a Linear width →
set to ego-motion dim), and extrinsics (`use_extrinsics=False`). What remains is a generic
action-conditioned spatiotemporal predictor. DINO-Foresight validated this paradigm on **Cityscapes
driving**; navigation's real differences (egomotion-induced global flow, new content at frame edges,
multi-modal agent futures) are data-distribution challenges for any feature WM, handled by
ego-motion conditioning + residual prediction and settled empirically by Phase 1's copy-baseline.

## Temporal mechanics (context & horizon)

Context = the variable-length frame sequence fed as `x` (T inferred from length; capped at
`num_frames // tubelet_size`, which you set). Each frame is a block `[action, state, H·W patches]`;
the block-causal mask lets every frame attend bidirectionally within itself and causally to **all**
past frames (full history, not a sliding window). No recurrent state — context is in-context tokens.
Per forward it predicts **one step ahead**; longer horizons come from **autoregressive rollout**
(possible because out-dim == in-dim). So both context length and horizon are **variable / your
choice**; the practical horizon limit is drift, addressed by multi-step rollout training + horizon
eval (1/5/10/20) vs the copy baseline.

## Phase 0 — Compression de-risk (gates everything) — DO FIRST

New: `src/jepa_drive_wm/world_model/pca_compression_derisk.py`.
Load `Outputs/vjepa21_depth/depth_probe_best.pt` (confirm it's the 4-layer DPT); val = KITTI seq 10;
online 4-layer encode via `VJEPA21Wrapper.encode_images_hierarchical` (no hier cache needed). Fit
per-layer PCA on a train sample; **sweep** dim {192,96,64} × layers {4,2}; reconstruct; run the
frozen DPT on original vs reconstructed; report AbsRel/RMSE/δ1 deltas
(`.../vjepa21_depth_simple/metrics.py`) + a few depth viz. Output: the smallest state within ~5–10%
AbsRel. If it collapses, escalate compressor (learned linear AE → task-supervised) before the WM.

## Phase 1 — KITTI depth-only world model (the proof, existing assets only)

KITTI gives consecutive frames + GT ego-motion (`gt_poses` → `relative_left_color_pose`) + a working
depth probe. No CARLA, no new GT, no sim.
1. Cache full-grid, channel-compressed per-frame latents for the KITTI sequences.
2. Retrain a small DPT depth probe to decode from the compressed latent (in_channels = chosen dim).
3. Train the tiny AC predictor from scratch: teacher-forced next-frame **residual**, ego-motion from
   relative poses, block-causal, per-patch MSE+cosine in latent space. T≈4–8, cached latents.
4. Eval decoded depth AbsRel at horizons 1/5/10/20, and it **must beat a copy baseline**
   (predict = last frame). Counterfactual: perturb the pose action → different imagined future.

New files: `world_model/ac_world_model.py` (wraps `vit_ac_predictor`), KITTI latent dataloader, WM
train script.

## Phase 2+ — CARLA multi-task (later extension, only after KITTI proof works)

Adds semantics/normals (CARLA GT), controlled/counterfactual actions, and per-task DPT probes. Reuse
`sim/collect_carla_sequence.py`. Same tiny/cached/full-grid recipe.

## Verification / success criteria

- Phase 0: orig-vs-reconstructed depth metric table + side-by-side viz; success = small AbsRel delta.
- Phase 1: decoded AbsRel vs horizon; **must beat the copy baseline**; counterfactual actions produce
  visibly different rollouts. Never judge on feature-MSE alone (hides rollout drift).

## Key files

- Reuse: `utils/vjepa_wrapper.py` (`encode_images_hierarchical`), `data/global_pca.py`,
  `probes/depth/vjepa21_depth_simple/{dpt_head,metrics,config,dataset}.py`, `data/kitti.py`,
  `src/models/ac_predictor.py`, `src/models/utils/modules.py` (ACBlock/RoPE), `sim/collect_carla_sequence.py`.
- New: `world_model/pca_compression_derisk.py` (Phase 0), `world_model/ac_world_model.py` +
  dataloader + train script (Phase 1).

## Open items (defaults chosen; revisit if needed)

- Hardware not specified → sizing left tunable via channel dim / T / batch+accum / checkpointing
  (grid fixed at full res). Start scope defaulted to **KITTI depth-only first**.
- Status: Phase 0 (compression de-risk) in progress.
