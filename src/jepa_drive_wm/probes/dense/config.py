"""Config for the single SFP dense probe, shared by depth and semantic segmentation.

One dataclass with a ``task`` selector. Shared fields (split, encoder, SFP dims, optimiser,
schedule, W&B) are identical across tasks; ``task`` picks the depth-only or semseg-only fields,
the loss recipe, and (via ``tasks.py``) the dataset / decode / metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

KITTI_SEQUENCES_DIR = "/home/hashim/Desktop/Datasets/KITTI/data_odometry_color/dataset/sequences"
SEMANTIC_ROOT = "/home/hashim/Desktop/Datasets/KITTI/semantic_oneformer"


@dataclass
class DenseConfig:
    task: str = "depth"                   # "depth" | "semseg"

    # --- data / split -------------------------------------------------------
    kitti_sequences_dir: str = KITTI_SEQUENCES_DIR
    image_dirname: str = "image_2"
    vjepa_size: str = "BASE"
    feat_mode: str = "final"              # "final" (768-d) | "pred" (384-d predictor_embed)
    train_sequences: tuple[int, ...] = (0, 1, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21)
    val_sequences: tuple[int, ...] = (2, 9)
    test_sequences: tuple[int, ...] = (10, 13)
    target_hw: tuple[int, int] = (384, 1248)
    image_hw: Optional[tuple[int, int]] = None   # encoder input res; defaults to target_hw
    augment: bool = True

    # --- head (SimpleFeaturePyramid) ---------------------------------------
    embed_dim: int = 768
    pred_embed_dim: int = 384
    sfp_channels: int = 256
    sfp_pyramid_channels: tuple[int, ...] = (128, 256, 512, 1024)

    # --- depth-only ---------------------------------------------------------
    n_bins: int = 256
    min_depth: float = 0.001
    max_depth: float = 80.0
    bins_strategy: str = "log"
    norm_strategy: str = "linear"
    bin_min_depth: float = 1.0
    bin_max_depth: float = 80.0
    eval_scale_align: bool = True
    depth_losses: dict = field(default_factory=lambda: {"SIGLOSS": 1.0, "GRADIENT_LOG_LOSS": 0.25, "L1": 0.1})

    # --- semseg-only --------------------------------------------------------
    semantic_root: str = SEMANTIC_ROOT
    num_classes: int = 19
    ignore_index: int = 255
    semseg_losses: dict = field(default_factory=lambda: {"CE": 1.0})

    # --- optimisation -------------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 35.0
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_workers: int = 4
    total_iters: int = 50000
    warmup_iters: int = 2000
    seed: int = 0

    # --- eval / logging -----------------------------------------------------
    eval_every: int = 2500
    eval_max_batches: int = 150
    log_every: int = 50
    out_dir: str = "/home/hashim/Desktop/Outputs/vjepa21_dense"

    # --- Weights & Biases ---------------------------------------------------
    wandb: bool = True
    wandb_project: str = "vjepa21-dense-probe"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: str = "sfp"
    wandb_num_images: int = 4

    # --- derived ------------------------------------------------------------
    @property
    def embed_dim_for_head(self) -> int:
        return self.pred_embed_dim if self.feat_mode == "pred" else self.embed_dim

    @property
    def n_out(self) -> int:
        return self.n_bins if self.task == "depth" else self.num_classes

    @property
    def losses(self) -> dict:
        return self.depth_losses if self.task == "depth" else self.semseg_losses
