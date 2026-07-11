# dpt_probe — depth from V-JEPA 2.1, final-layer vs intermediate layers

Dense metric depth from a **DPT head**, used to run one controlled experiment:

> Does depth need the **intermediate** V-JEPA 2.1 layers, or is the **final** layer enough?

Same head, same training, same encoder pass — the **only** variable is which encoder
outputs feed the head's 4 input slots (`config.layer_mode`):

| `layer_mode` | fed to the DPT head | dim | question it tests |
|---|---|---|---|
| `"quad"` | `[L3, L6, L9, L12]` | 768 | the DINOv3-style intermediate-layer hierarchy |
| `"final"` | `[L12, L12, L12, L12]` | 768 | V-JEPA 2.1's claim: the final layer alone suffices |
| `"pred"` | `[P, P, P, P]`, `P = predictor_embed(L12)` | 384 | is a native learned 2× compression still depth-decodable? |

Because the decoder is identical (only its input-conv width changes: 768 vs 384 for `pred`),
any performance gap is attributable to the features, not the head.

- `quad` vs `final` tests V-JEPA 2.1 §3.5's claim ("we do not utilize intermediate layers")
  on our *distilled* ViT-B, whose intermediate layers never got deep self-supervision.
- `pred` tests whether the predictor's own input projection `predictor_embed` (a learned
  `Linear(768→384)`, the compression from the paper's Figure 4) preserves depth — i.e. a free
  2× smaller candidate **world-model state**. Note `predictor_embed(L12)` is a compressed
  representation of the *current* frame, **not** a forward-predicted one.

## Pipeline

```
KITTI image  (3, H, W)
  → ImageDepthDataset               → (B, 3, image_hw)     preprocessed, ImageNet-normalised
  → frozen V-JEPA encoder (ONLINE)  → (B, 4, D=768, 24, 78)  the 4 taps [L3,L6,L9,L12]
  → select by layer_mode            → quad: as-is | final: L12 repeated 4x
  → DPTDepthHead (reassemble → fuse)→ (B, n_bins, 384, 1248)  logits
  → binning (AdaBins soft-argmax)   → (B, 1, 384, 1248)  metric depth
  → SigLoss
```

### How DPT gets sharpness (`dpt_head.py`)

1. **Reassemble** — each of the 4 inputs is resampled to a *different* resolution (shallow
   slot upsampled 4×, deepest downsampled), forming a feature pyramid. In `final` mode the
   pyramid is built from the same L12 four times — a standard single-map DPT.
2. **Fuse** — coarse-to-fine: interpolate ×2, add the next-finer level, residual conv.
3. **Decode** — `n_bins` logits → depth via the shared soft-argmax.

V-JEPA has no CLS token, so `dpt_readout="ignore"`.

## Feature source — **online only**

The 4-layer cache would need ~1 TB (the 1-layer cache is already ~286 GB; ~189 GB free), so
the frozen encoder runs **in the training loop** (`wrapper.extract_hierarchical`). Both modes
share the same forward pass — `final` just indexes the last layer — so the comparison is
never confounded by a cached-vs-online difference.

## Memory

DPT emits `n_bins` channels at full 384×1248 → heavy activations. Defaults target an ~8 GB
GPU: `batch_size=1`, `grad_accum_steps=4` (effective batch 4). Run with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Run the experiment

```bash
# arm 1: intermediate-layer hierarchy
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m jepa_drive_wm.probes.depth.dpt_probe.train --mode quad

# arm 2: final layer only
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m jepa_drive_wm.probes.depth.dpt_probe.train --mode final

# arm 3: predictor_embed 384-d compression
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m jepa_drive_wm.probes.depth.dpt_probe.train --mode pred
```

Each writes to `~/Desktop/Outputs/vjepa21_depth_dpt/<mode>/depth_probe_best.pt` (separate
sub-dirs, so the two arms never clobber). Compare the final val `abs_rel` / `a1`:
- `final` ≈ `quad` → the final representation is enough → the world model can use it.
- `quad` ≫ `final` → intermediate layers carry real dense info → the L3-state + hierarchy
  route is justified.

## Files

| file          | what |
|---------------|------|
| `config.py`   | `DPTProbeConfig` — knobs, incl. `layer_mode` |
| `dataset.py`  | `ImageDepthDataset` — raw images + depth for online encoding |
| `dpt_head.py` | the DPT reassemble/fusion internals (DINOv3 protocol) |
| `head.py`     | `DPTDepthHead`, `DptDepthProbe` |
| `train.py`    | online training loop, layer selection, optional curriculum, eval |
| `visualize.py`| RGB / GT / predicted-depth triptych for a checkpoint |

Shared machinery in [`../_core`](../_core): losses, metrics, depth binning, KITTI loading.
