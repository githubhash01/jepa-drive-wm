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
    # Feature source. "cached" reads precomputed .npy (embedding_dirname); "online"
    # encodes each batch through the frozen VJEPA encoder in the train loop (no disk
    # cache — use this when the 4-layer cache is too big for your SSD). For the DPT head,
    # "online" requires nothing on disk beyond the KITTI images themselves.
    feature_mode: str = "cached"          # "cached" or "online"
    image_dirname: str = "image_2"        # online mode: KITTI camera folder to encode
    vjepa_size: str = "BASE"              # online mode: VJEPA21Size name of the frozen encoder
    # Input image resolution fed to the encoder (online mode). Defaults to target_hw.
    # Depth is always supervised at target_hw, so this only changes the patch grid.
    image_hw: Optional[tuple[int, int]] = None
    # Two-phase / curriculum training (DINO-Foresight resolution adaptation): a list of
    # per-stage dicts, e.g. [{"image_hw": (192, 624), "iters": 28000, "lr": 3e-4,
    # "warmup_iters": 1000}, {"image_hw": (384, 1248), "iters": 12000, "lr": 1e-4,
    # "warmup_iters": 500}]. The SAME probe (fully-conv, so weights transfer across grids)
    # is carried across stages. If None, a single stage is built from total_iters/lr below.
    stages: Optional[list] = None
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
    augment: bool = False                # online mode: random h-flip + colour jitter on the
                                         # train split (re-encoded each epoch -> free aug)

    # --- depth parameterisation --------------------------------------------
    n_bins: int = 256                    # output channels of the linear head
    min_depth: float = 0.001             # valid-mask floor (permissive; masks depth <= this)
    max_depth: float = 80.0              # valid-mask / clamp ceiling
    bins_strategy: str = "log"           # "log", "linear", or "mixlog" (DINOv3 log/linear blend)
    norm_strategy: str = "linear"        # AdaBins soft-argmax normalisation
    # Soft-argmax bin range, DECOUPLED from the valid-mask floor above. KITTI depth is
    # ~2.2-80m, so the default [0.001, 80] wasted ~67% of log-bins below the nearest real
    # pixel. Spanning the actual range concentrates all bins where depth occurs (DORN/SID
    # + AdaBins principle: match the bin distribution to the dataset).
    bin_min_depth: float = 1.0
    bin_max_depth: float = 80.0
    # True adaptive bins (AdaBins): an MLP predicts per-image bin centers within
    # [bin_min_depth, bin_max_depth] instead of using fixed ones. bin_chamfer_weight adds
    # the AdaBins bi-directional Chamfer bin-density regulariser so the bins actually adapt.
    adaptive_bins: bool = False
    bin_chamfer_weight: float = 0.1

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
    grad_accum_steps: int = 1            # accumulate this many micro-batches per optimiser step
                                         # (effective batch = batch_size * grad_accum_steps).
                                         # Lets bs=1 DPT reach a larger effective batch on 8 GB.
    num_workers: int = 4
    total_iters: int = 8000              # micro-batches (frames seen); optimiser updates = /accum
    warmup_iters: int = 200              # linear LR warm-up before cosine decay (micro-batch units)
    warm_up_loss: bool = True            # SigLoss scale warm-up

    # --- eval / logging -----------------------------------------------------
    eval_every: int = 500
    eval_max_batches: int = 150          # periodic eval samples this many val batches
                                         # (the final eval at total_iters runs the full set)
    eval_scale_align: bool = False       # also report median-scaled metrics (abs_rel_al / a1_al);
                                         # SigLoss is scale-invariant, so this isolates structure
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
