"""Configuration for the V-JEPA 2.1 linear depth probe."""
from __future__ import annotations

from dataclasses import dataclass, field


# Where the cached embeddings and KITTI depth PNGs live (see data/kitti.py for the
# KITTI layout). Sequence dirs are named e.g. ``seq00_vjepa21_large``.
EMBEDDINGS_DIR = "/home/hashim/Desktop/Datasets/ImageEmbeddings"
KITTI_SEQUENCES_DIR = "/home/hashim/Desktop/Datasets/KITTI/data_odometry_color/dataset/sequences"
EMBEDDING_MODEL = "vjepa21_large"


@dataclass
class DepthProbeConfig:
    # --- data ---------------------------------------------------------------
    embeddings_dir: str = EMBEDDINGS_DIR
    kitti_sequences_dir: str = KITTI_SEQUENCES_DIR
    embedding_model: str = EMBEDDING_MODEL
    # Only seqs 00/02/10 have FoundationStereo depth. Sequence-level holdout.
    train_sequences: tuple[int, ...] = (0, 2)
    val_sequences: tuple[int, ...] = (10,)
    ram_cache: bool = False              # keep bf16 feature tensors in RAM (~7 GB for 00/02/10)

    # --- depth parameterisation --------------------------------------------
    n_bins: int = 256                    # output channels of the linear head
    min_depth: float = 0.001
    max_depth: float = 80.0
    bins_strategy: str = "log"           # "log" or "linear"
    norm_strategy: str = "linear"        # AdaBins soft-argmax normalisation

    # --- head ---------------------------------------------------------------
    embed_dim: int = 1024                # V-JEPA 2.1 LARGE
    use_batchnorm: bool = True

    # --- optimisation -------------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 35.0
    batch_size: int = 8
    num_workers: int = 4
    total_iters: int = 8000
    warm_up_loss: bool = True            # SigLoss scale warm-up

    # --- eval / logging -----------------------------------------------------
    eval_every: int = 500
    log_every: int = 50
    seed: int = 0
    out_dir: str = "/home/hashim/Desktop/Outputs/vjepa21_depth"

    losses: dict = field(default_factory=lambda: {"SIGLOSS": 1.0})
