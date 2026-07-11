# depth — frozen-feature depth probes on V-JEPA 2.1

Dense depth from **frozen** V-JEPA 2.1 (ViT-B/16) features. The encoder never trains; only a
small DPT head does. This exists to answer one question for the world model:

> *Does good depth need the intermediate encoder layers, or is the final representation enough?*

The answer decides the world-model state. If the **final** representation suffices, the WM can
predict/decode it directly and stay simple. If the **intermediate layers** matter, the WM must
reconstruct the hierarchy (which predicting early-layer L3 + running the frozen encoder tail
does for free).

V-JEPA 2.1's paper (§3.5) claims the final layer *is* enough for dense tasks ("we do not
utilize intermediate layers"), thanks to deep self-supervision during pretraining. But our
checkpoint is a **distilled ViT-B** whose intermediate layers never got that supervision — so
we test the claim directly rather than assume it.

## The experiment — [`dpt_probe/`](dpt_probe)

One DPT head, one training pipeline, one config flag `layer_mode`:

| `layer_mode` | features into the head | dim | tests |
|---|---|---|---|
| `quad` | `[L3, L6, L9, L12]` | 768 | intermediate-layer hierarchy (DINOv3-style) |
| `final` | `[L12, L12, L12, L12]` | 768 | final layer only (V-JEPA 2.1's claim) |
| `pred` | `predictor_embed(L12) ×4` | 384 | native learned 2× compression as a WM state |

Same head, same loop, same online encoder pass — only the input features differ, so the
result is a clean read on what the world model can afford to use as its state. Run the arms:

```bash
python -m jepa_drive_wm.probes.depth.dpt_probe.train --mode quad
python -m jepa_drive_wm.probes.depth.dpt_probe.train --mode final
python -m jepa_drive_wm.probes.depth.dpt_probe.train --mode pred
```

Compare val `abs_rel` / `a1` between `~/Desktop/Outputs/vjepa21_depth_dpt/{quad,final,pred}/`:
`quad`≈`final` → the final layer suffices; `pred`≈`final` → the 384-d compression is free and
a smaller WM state is viable.

## [`_core/`](_core) — shared, probe-agnostic

The reusable depth machinery (also for the planned semantic-segmentation probe):

- `losses.py` — SigLoss & friends, `build_loss`, `chamfer_bin_loss`.
- `metrics.py` — `calculate_depth_metrics` (a1/a2/a3/abs_rel/rmse/silog).
- `binning.py` — logits → metric depth (`FeaturesToDepth`, `AdaptiveBins`, soft-argmax).
- `kitti.py` — depth PNG loading, valid mask, resize, batch collate.
- `viz.py` — RGB / GT / predicted-depth triptych.

(`dinov3_depth/` is the vendored DINOv3 reference implementation these were adapted from —
left as-is for reference.)
