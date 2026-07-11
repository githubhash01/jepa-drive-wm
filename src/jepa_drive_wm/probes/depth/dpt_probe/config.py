"""Config for the DPT depth probe (the final-vs-intermediate-layers experiment).

One DPT head, two feature inputs selected by ``layer_mode``:

* ``"quad"``  — feed the 4 tapped layers ``[L3, L6, L9, L12]`` (DINOv3-style hierarchy).
* ``"final"`` — feed the final layer ``L12`` repeated 4x (V-JEPA 2.1's claim: the final
                layer alone is enough for dense tasks). Same head, same everything else.

The **only** difference between the two runs is which encoder outputs go into the head, so
any gap is attributable to the intermediate layers, not the decoder. Features are always
produced **online** (the frozen encoder runs in the train loop) -- both modes share the
same encoder forward pass -- because the 4-layer cache is far too big for disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

KITTI_SEQUENCES_DIR = "/home/hashim/Desktop/Datasets/KITTI/data_odometry_color/dataset/sequences"


@dataclass
class DPTProbeConfig:
    # --- data ---------------------------------------------------------------
    kitti_sequences_dir: str = KITTI_SEQUENCES_DIR
    image_dirname: str = "image_2"       # KITTI camera folder to encode
    vjepa_size: str = "BASE"             # VJEPA21Size name of the frozen encoder (vit_base, D=768)
    # Explicit, disjoint sequence split over all 22 KITTI odometry seqs. NO overlap:
    #   train -> gradient updates only
    #   val   -> periodic eval + "best" checkpoint selection (model selection)
    #   test  -> final [test] metrics + visualization ONLY (never seen in train/selection)
    train_sequences: tuple[int, ...] = (0, 1, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21)
    val_sequences: tuple[int, ...] = (2, 9)
    test_sequences: tuple[int, ...] = (10, 13)
    target_hw: tuple[int, int] = (384, 1248)   # depth is always supervised at this resolution
    # Which encoder outputs feed the DPT head -- the one variable in the experiment.
    #   "quad"  -> [L3, L6, L9, L12]        (768-d each; intermediate-layer hierarchy)
    #   "final" -> [L12, L12, L12, L12]     (768-d; final layer only, repeated to the 4 slots)
    #   "pred"  -> [P, P, P, P]             (384-d; P = predictor_embed(L12), a learned 2x
    #                                        compression of the final layer, repeated to 4 slots)
    layer_mode: str = "quad"
    # Decoder head: "dpt" = 4-layer DPT reassembly (25M params); "linear" = DINOv3-style dense
    # linear probe (BatchNorm -> 1x1 conv -> soft-argmax, ~198k params) over the single final
    # grid. "linear" only makes sense with layer_mode in {final, pred} (it is single-layer).
    head_type: str = "dpt"
    augment: bool = True                 # random gamma + grayscale + h-flip on the train split
    # Input resolution fed to the encoder. Depth stays at target_hw, so this only changes the
    # patch grid. Defaults to target_hw when None.
    image_hw: Optional[tuple[int, int]] = None
    # Optional curriculum (DINO-Foresight resolution adaptation): a list of per-stage dicts,
    # e.g. [{"image_hw": (192, 624), "iters": 6000, "lr": 3e-4, "warmup_iters": 500},
    #       {"image_hw": (384, 1248), "iters": 4000, "lr": 1e-4, "warmup_iters": 200}].
    # The fully-conv DPT head transfers across grids. If None, one stage from the flat fields.
    stages: Optional[list] = None

    # --- depth parameterisation --------------------------------------------
    n_bins: int = 256
    min_depth: float = 0.001             # valid-mask floor
    max_depth: float = 80.0              # valid-mask / clamp ceiling
    bins_strategy: str = "log"           # "log", "linear", or "mixlog"
    norm_strategy: str = "linear"        # soft-argmax normalisation
    bin_min_depth: float = 1.0           # soft-argmax range (KITTI ~2.2-80m)
    bin_max_depth: float = 80.0

    # --- head ---------------------------------------------------------------
    embed_dim: int = 768                 # V-JEPA 2.1 BASE final/intermediate layer width
    pred_embed_dim: int = 384            # predictor_embed output width (used only by layer_mode="pred")
    n_layers: int = 4                    # tapped layers ([2,5,8,11]); DPT reassemble is fixed to 4
    dpt_channels: int = 256              # common channel width inside the DPT fusion path
    dpt_post_process_channels: tuple[int, ...] = (128, 256, 512, 1024)
    dpt_readout: str = "ignore"          # V-JEPA has no CLS token -> ignore readout
    dpt_use_batchnorm: bool = False      # keep off SyncBatchNorm (needs a process group)

    # --- optimisation -------------------------------------------------------
    # DPT emits n_bins channels at full 384x1248 res -> heavy activations. On an ~8 GB GPU
    # use batch_size=1 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and lean on
    # grad_accum_steps for a larger effective batch; larger GPUs can raise batch_size.
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 35.0
    batch_size: int = 1
    grad_accum_steps: int = 8            # eff_batch = batch_size * grad_accum_steps = 8
    num_workers: int = 4
    total_iters: int = 50000             # frames seen; optimiser updates = /grad_accum_steps
    warmup_iters: int = 2000
    seed: int = 0

    # --- eval / logging -----------------------------------------------------
    eval_every: int = 2500
    eval_max_batches: int = 150
    eval_scale_align: bool = True        # report metric (non-aligned) AND median-scale-aligned
    log_every: int = 50
    out_dir: str = "/home/hashim/Desktop/Outputs/vjepa21_depth_dpt_metric"

    # --- Weights & Biases (on by default; graceful no-op if wandb login is missing) ---------
    wandb: bool = True
    wandb_project: str = "vjepa21-depth-probe"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None      # defaults to layer_mode
    wandb_group: str = "layer-ablation"       # overlays quad/final/pred in the W&B UI
    wandb_num_images: int = 4                 # test-frame triptychs logged per eval (0 = off)

    # Metric depth recipe (lift-splat needs true meters):
    #   SIGLOSS       — SILog(λ=0.85), the AdaBins/DPT/ZoeDepth standard (primary).
    #   GRADIENT_LOG  — multi-scale gradient matching (MiDaS/DPT), sharpens depth edges.
    #   L1            — light metric anchor so absolute Z is trustworthy.
    losses: dict = field(default_factory=lambda: {"SIGLOSS": 1.0, "GRADIENT_LOG_LOSS": 0.25, "L1": 0.1})

    @property
    def head_embed_dim(self) -> int:
        """Per-layer channel width the DPT head is built for (384 in 'pred' mode, else 768)."""
        return self.pred_embed_dim if self.layer_mode == "pred" else self.embed_dim
