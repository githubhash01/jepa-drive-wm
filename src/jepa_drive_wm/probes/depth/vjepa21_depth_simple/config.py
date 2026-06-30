"""Configuration for the V-JEPA 2.1 linear depth probe."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# The cached V-JEPA embeddings now live *inside* each KITTI sequence dir, next to
# ``image_2`` / ``depth`` (written by utils/vjepa_embeddings_builder.py):
#
#     sequences/NN/vjepa_vitb/000000.npy   (grid_h*grid_w, embed_dim) fp16
#     sequences/NN/vjepa_vitb/_metadata.json
#
# so there is no separate embeddings directory any more.
KITTI_SEQUENCES_DIR = "/home/hashim/Desktop/Datasets/KITTI/data_odometry_color/dataset/sequences"
EMBEDDING_DIRNAME = "vjepa_vitb"          # smallest V-JEPA 2.1 model (vit_base, D=768)


@dataclass
class DepthProbeConfig:
    # --- data ---------------------------------------------------------------
    kitti_sequences_dir: str = KITTI_SEQUENCES_DIR
    embedding_dirname: str = EMBEDDING_DIRNAME
    # All 22 sequences have both depth + embeddings. Sequence-level holdout:
    # val_sequences are held out; ``train_sequences`` (property below) is then
    # "everything else". To pin a custom train set, set ``train_override``.
    val_sequences: tuple[int, ...] = (10,)
    train_override: Optional[tuple[int, ...]] = None
    all_sequences: tuple[int, ...] = tuple(range(22))
    # KITTI native depth resolutions still differ slightly across sequences
    # (376x1241 / 375x1242 / 370x1226), so resize depth + mask to this fixed,
    # patch-aligned (H, W) for cross-sequence batching and aligned supervision.
    # 384x1248 ~= native and matches the encoder input the embeddings used.
    target_hw: tuple[int, int] = (384, 1248)
    ram_cache: bool = False              # keep fp16 feature tensors in RAM (~3 MB/frame)

    # --- depth parameterisation --------------------------------------------
    n_bins: int = 256                    # output channels of the linear head
    min_depth: float = 0.001
    max_depth: float = 80.0
    bins_strategy: str = "log"           # "log" or "linear"
    norm_strategy: str = "linear"        # AdaBins soft-argmax normalisation

    # --- head ---------------------------------------------------------------
    embed_dim: int = 768                 # V-JEPA 2.1 BASE (vit_base)
    use_batchnorm: bool = True
    head_type: str = "conv"              # "conv", "linear", or "dpt" (4-layer DPT)
    decoder_channels: int = 256          # conv decoder width at the patch grid
    decoder_upsample: int = 3            # conv decoder 2x upsample steps: 24x78 -> 192x624

    # --- dpt head (only used when head_type == "dpt") -----------------------
    # Fuses 4 intermediate VJEPA layers ([2,5,8,11]) DINOV3-style. Requires the
    # hierarchical cache, so point ``embedding_dirname`` at "vjepa_vitb_hier".
    n_layers: int = 4                    # number of tapped layers in the hierarchical cache
    dpt_channels: int = 256              # common channel width inside the DPT fusion path
    dpt_post_process_channels: tuple[int, ...] = (128, 256, 512, 1024)
    dpt_readout: str = "ignore"          # VJEPA has no CLS token -> ignore readout
    dpt_use_batchnorm: bool = False      # keep off SyncBatchNorm (needs a process group)
    # DPT emits n_bins channels at full 384x1248 res -> heavy activations. On an ~8 GB
    # GPU use batch_size=1 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # (batch_size>=2 OOMs); larger GPUs can raise it.

    # --- optimisation -------------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 35.0
    batch_size: int = 4                  # conv decoder's high-res bins are memory-heavy on ~8 GB GPUs
    num_workers: int = 4
    total_iters: int = 8000
    warmup_iters: int = 200              # linear LR warm-up before cosine decay
    warm_up_loss: bool = True            # SigLoss scale warm-up

    # --- eval / logging -----------------------------------------------------
    eval_every: int = 500
    log_every: int = 50
    seed: int = 0
    out_dir: str = "/home/hashim/Desktop/Outputs/vjepa21_depth"

    losses: dict = field(default_factory=lambda: {"SIGLOSS": 1.0})

    @property
    def train_sequences(self) -> tuple[int, ...]:
        """Train split = ``train_override`` if set, else all sequences minus val."""
        if self.train_override is not None:
            return self.train_override
        val = set(self.val_sequences)
        return tuple(s for s in self.all_sequences if s not in val)
