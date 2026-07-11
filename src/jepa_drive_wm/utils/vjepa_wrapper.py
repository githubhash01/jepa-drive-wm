"""
VJEPA21Wrapper — a thin, explicit interface over a local V-JEPA 2.1 checkout.

What this module gives you
--------------------------
* One enum (`VJEPA21Size`) that ties together architecture name, checkpoint
  file and embed dim, so size/checkpoint can never disagree.
* An encoder that can embed single images, batches of images, and video
  clips (multi-frame), on cpu or cuda, with sane dtype handling.
* A `TokenLayout` describing how flat tokens map back to (T', H', W') grids,
  shared by all outputs of a given wrapper — instead of repeating grid
  metadata inside every embedding object.

Typical usage
-------------
    wrapper = VJEPA21Wrapper(size=VJEPA21Size.LARGE)

    feats = wrapper.encode_images(seq.left_images[:8])      # (8, Hp*Wp, D)
    clip  = wrapper.encode_clip(seq.left_images[:16])       # (1, T'·Hp·Wp, D)

    layout = wrapper.layout(num_frames=16)
    grid   = wrapper.unflatten(clip, num_frames=16)         # (1, T', Hp, Wp, D)

    # predict the last temporal slice from the first seven
    ctx, tgt = wrapper.temporal_slice_masks(range(0, 7), [7], num_frames=16)
    wrapper.load_predictor()
    pred = wrapper.predict(clip_context_tokens, ctx, tgt)   # see predict() docs
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
from PIL import Image

import torch
import numpy as np
from torchvision import transforms

# Hardcoded local paths — keep it simple.
VJEPA_REPO = "/home/hashim/Modules/vjepa2"
VJEPA_CHECKPOINTS_DIR = "/home/hashim/Modules/model_checkpoints/vjepa21"

# Import locations inside the V-JEPA2 repo. Kept in one place so a repo
# re-organisation costs you exactly two lines.
ENCODER_MODULE = "app.vjepa_2_1.models.vision_transformer"

ImageLike = Union[str, Path, Image.Image, torch.Tensor, np.ndarray]


class VJEPA21Size(Enum):
    """Architecture + checkpoint + embed dim, bound together."""

    BASE = ("vit_base", "vjepa2_1_vitb_dist_vitG_384.pt", 768)
    LARGE = ("vit_large", "vjepa2_1_vitl_dist_vitG_384.pt", 1024)
    GIANT = ("vit_giant_xformers", "vjepa2_1_vitg_384.pt", 1408)
    GIGANTIC = ("vit_gigantic_xformers", "vjepa2_1_vitG_384.pt", 1664)

    @property
    def arch(self) -> str:
        return self.value[0]

    @property
    def checkpoint_name(self) -> str:
        return self.value[1]

    @property
    def embed_dim(self) -> int:
        return self.value[2]


@dataclass(frozen=True)
class TokenLayout:
    """How a flat (N_tokens, D) sequence maps back onto space-time.

    One layout describes *every* clip of the same length produced by the same
    wrapper — it is not duplicated per frame/per embedding.
    """

    num_frames: int          # input frames T (1 for single images)
    tubelet_size: int        # temporal patch size (2 for V-JEPA 2.1 video mode)
    grid_t: int              # temporal token slices T' = T / tubelet (1 for images)
    grid_h: int              # token rows   = img_height / patch_size
    grid_w: int              # token cols   = img_width  / patch_size
    patch_size: int
    embed_dim: int

    @property
    def tokens_per_slice(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def num_tokens(self) -> int:
        return self.grid_t * self.tokens_per_slice

    def slice_token_range(self, t: int) -> range:
        """Flat token indices belonging to temporal slice `t` (0-based)."""
        if not 0 <= t < self.grid_t:
            raise IndexError(f"slice {t} out of range [0, {self.grid_t})")
        start = t * self.tokens_per_slice
        return range(start, start + self.tokens_per_slice)


class VJEPA21Wrapper:
    """Encoder (+ optional predictor) over a local V-JEPA 2.1 checkout."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        size: VJEPA21Size = VJEPA21Size.BASE,
        device: Optional[str] = None,
        image_height: int = 384,
        image_width: int = 1248,
        num_frames: int = 64,
        tubelet_size: int = 2,
        patch_size: int = 16,
        encoder_key: str = "ema_encoder",
        checkpoint_path: Optional[Union[str, Path]] = None,
        repo_path: Optional[Union[str, Path]] = None,
        compute_dtype: Optional[torch.dtype] = None,
        output_dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ) -> None:
        self.size = size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.img_height = image_height
        self.img_width = image_width
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.encoder_key = encoder_key
        self.verbose = verbose

        # bf16 autocast on cuda, plain fp32 on cpu (autocast 'cuda' would crash there).
        if compute_dtype is None:
            compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype

        self.repo_path = Path(
            repo_path if repo_path is not None else VJEPA_REPO
        ).expanduser().resolve()

        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else Path(VJEPA_CHECKPOINTS_DIR).expanduser().resolve() / self.size.checkpoint_name
        )

        
        self._checkpoint_blob: Optional[dict] = None   # lazy, freeable cache
        self._predictor_embed: Optional[torch.nn.Linear] = None  # lazy predictor input projection

        # Built once; the original rebuilt this per embed_image call.
        self._preprocess = transforms.Compose(
            [
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

        self.encoder = self._build_encoder()

    def _ensure_vjepa_repo_on_path(self) -> None:
        expected = self.repo_path / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"

        if not expected.is_file():
            raise FileNotFoundError(
                f"VJEPA_REPO should point at the root of the V-JEPA2 checkout, "
                f"but I could not find: {expected}"
            )

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _read_checkpoint(self) -> dict:
        if self._checkpoint_blob is None:
            self._checkpoint_blob = torch.load(
                self.checkpoint_path, weights_only=True, map_location="cpu"
            )
            if self.verbose:
                print(f"[VJEPA21] loaded checkpoint {self.checkpoint_path.name}, "
                      f"keys: {list(self._checkpoint_blob.keys())}")
        return self._checkpoint_blob

    def free_checkpoint_cache(self) -> None:
        """Drop the raw checkpoint blob (call after loading encoder+predictor)."""
        self._checkpoint_blob = None

    @staticmethod
    def _clean_state_dict(state: dict) -> dict:
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        return state

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_encoder(self) -> torch.nn.Module:
        self._ensure_vjepa_repo_on_path()

        import importlib
        vit_module = importlib.import_module(ENCODER_MODULE)
        factory = getattr(vit_module, self.size.arch)

        encoder = factory(
            patch_size=self.patch_size,
            img_size=(self.img_height, self.img_width),
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,   # B C 1 H W routes to image patch-embed
            interpolate_rope=True,
        )

        blob = self._read_checkpoint()
        if self.encoder_key not in blob:
            raise KeyError(
                f"'{self.encoder_key}' not in checkpoint; available: {list(blob.keys())}"
            )
        state = self._clean_state_dict(blob[self.encoder_key])
        msg = encoder.load_state_dict(state, strict=True)
        if self.verbose:
            print(f"[VJEPA21] encoder {self.size.arch} <- '{self.encoder_key}': {msg}")

        encoder = encoder.to(self.device)
        if self.device.type == "cuda":
            encoder = encoder.to(self.compute_dtype)
        encoder.eval()
        return encoder

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def prepare_frame(self, image: ImageLike) -> torch.Tensor:
        """-> normalized (C, H, W) float tensor at the wrapper's resolution."""
        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(image.shape)}")
            return image  # assume caller already preprocessed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image).convert("RGB")
            elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
                if image.shape[2] == 1:
                    image = image[:, :, 0]
                image = Image.fromarray(image).convert("RGB")
            else:
                raise ValueError(f"Expected HxW, HxWx1, HxWx3, or HxWx4 ndarray, got {image.shape}")
        return self._preprocess(image)

    def _stack_frames(self, frames: Sequence[ImageLike]) -> torch.Tensor:
        return torch.stack([self.prepare_frame(f) for f in frames], dim=0)  # (T,C,H,W)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.compute_dtype if self.device.type == "cuda" else None)
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                out = self.encoder(x)
        else:
            out = self.encoder(x)
        return out.to("cpu", self.output_dtype)

    @torch.inference_mode()
    def _run_encoder_multi(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder in multi-layer mode and stack the per-layer outputs.

        Requires ``self.encoder.out_layers`` to be set (done by
        ``_enable_hierarchical``), which makes ``VisionTransformer.forward`` return a
        ``list`` of per-layer normed token tensors (each ``(B, N, D)``) instead of a
        single final-layer tensor. Returns ``(B, L, N, D)`` on cpu.
        """
        x = x.to(self.device, self.compute_dtype if self.device.type == "cuda" else None)
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                outs = self.encoder(x)
        else:
            outs = self.encoder(x)
        if not isinstance(outs, (list, tuple)):
            raise RuntimeError(
                "Expected a list of per-layer outputs from the encoder; got a single "
                "tensor. Was encoder.out_layers set? (call _enable_hierarchical first)"
            )
        stacked = torch.stack(list(outs), dim=1)  # (B, L, N, D)
        return stacked.to("cpu", self.output_dtype)

    def _enable_hierarchical(self) -> list[int]:
        """Put the encoder into multi-layer output mode and return the tapped layers.

        Sets ``encoder.out_layers`` to the model's predefined ``hierarchical_layers``
        ([2,5,8,11] for ViT-B). Each tapped block is normed by its own ``norms_block``
        entry. Idempotent.
        """
        layers = list(self.encoder.hierarchical_layers)
        if getattr(self.encoder, "out_layers", None) != layers:
            self.encoder.out_layers = layers
        return layers

    @property
    def hierarchical_layers(self) -> list[int]:
        return list(self.encoder.hierarchical_layers)

    def encode_images(
        self, images: Union[ImageLike, Sequence[ImageLike]], batch_size: int = 8
    ) -> torch.Tensor:
        """Encode frames *independently* (image mode, T=1 each).

        Returns (N, grid_h * grid_w, embed_dim) on cpu — one token grid per
        frame, no temporal mixing. Pass a single image to get N == 1.
        """
        if isinstance(images, (str, Path, Image.Image, torch.Tensor)):
            images = [images]
        chunks = []
        for i in range(0, len(images), batch_size):
            batch = self._stack_frames(images[i : i + batch_size])     # (B,C,H,W)
            batch = batch.unsqueeze(2)                                 # (B,C,1,H,W)
            chunks.append(self._run_encoder(batch))
        return torch.cat(chunks, dim=0)

    def encode_clip(self, frames: Sequence[ImageLike]) -> torch.Tensor:
        """Encode frames *jointly* as a video clip (temporal attention).

        len(frames) must be a multiple of tubelet_size (2). Returns
        (1, (T / tubelet) * grid_h * grid_w, embed_dim) on cpu.
        """
        T = len(frames)
        if T < self.tubelet_size or T % self.tubelet_size != 0:
            raise ValueError(
                f"Clip length must be a positive multiple of tubelet_size="
                f"{self.tubelet_size}, got {T}"
            )
        clip = self._stack_frames(frames)            # (T,C,H,W)
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1,C,T,H,W)
        return self._run_encoder(clip)

    @torch.no_grad()
    def extract_hierarchical(self, batch_chw: torch.Tensor) -> torch.Tensor:
        """Frozen 4-layer feature extraction for *online* training (no disk cache).

        ``batch_chw``: an already-preprocessed ``(B, C, H, W)`` batch (resized to the
        wrapper's resolution + ImageNet-normalised, e.g. by ``ImageDepthDataset``).
        Returns ``(B, L, D, grid_h, grid_w)`` on this wrapper's device, ready to feed the
        depth head — like a batch of the hierarchical cache but computed on the fly.

        Uses ``torch.no_grad`` (not ``inference_mode``): the encoder is frozen so we want
        no gradients through it, but the output must still be usable as input to a
        *trainable* head, which inference-mode tensors are not.
        """
        self._enable_hierarchical()
        x = batch_chw.to(self.device, self.compute_dtype if self.device.type == "cuda" else None)
        x = x.unsqueeze(2)  # (B,C,H,W) -> (B,C,1,H,W) routes to the image patch-embed
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                outs = self.encoder(x)
        else:
            outs = self.encoder(x)
        if not isinstance(outs, (list, tuple)):
            raise RuntimeError("Expected a list of per-layer outputs; was out_layers set?")
        stacked = torch.stack(list(outs), dim=1)  # (B, L, N, D)
        B, L, _, D = stacked.shape
        lay = self.layout(num_frames=1)
        feat = stacked.reshape(B, L, lay.grid_h, lay.grid_w, D).permute(0, 1, 4, 2, 3).contiguous()
        return feat.float()

    # ------------------------------------------------------------------
    # Predictor input projection (a compressed state)
    # ------------------------------------------------------------------
    def _load_predictor_embed(self) -> torch.nn.Linear:
        """Build the predictor's input projection ``predictor_embed`` as a standalone Linear.

        In the pretraining/distillation predictor, ``x = predictor_embed(x)`` is the FIRST
        op — a linear projection of the encoder output before any masked prediction. For our
        distilled ViT-B it is a single ``Linear(embed_dim -> predictor_embed_dim)`` (e.g.
        768 -> 384). The multi-level fusion variant (``Sequential`` over concatenated layers)
        exists only in deep-supervised teacher checkpoints, which we don't load here.
        """
        blob = self._read_checkpoint()
        if "predictor" not in blob:
            raise KeyError("checkpoint has no 'predictor' weights; cannot expose predictor_embed")
        state = self._clean_state_dict(blob["predictor"])
        w = state.get("predictor_embed.weight")
        b = state.get("predictor_embed.bias")
        if w is None or w.ndim != 2:
            raise RuntimeError(
                "predictor_embed is not a single Linear in this checkpoint "
                f"(weight={None if w is None else tuple(w.shape)}). The multi-level fusion MLP "
                "(a Sequential) only exists in deep-supervised teacher checkpoints (e.g. ViT-G)."
            )
        out_dim, in_dim = w.shape
        lin = torch.nn.Linear(in_dim, out_dim, bias=b is not None)
        with torch.no_grad():
            lin.weight.copy_(w)
            if b is not None:
                lin.bias.copy_(b)
        lin = lin.to(self.device).eval()
        for p in lin.parameters():
            p.requires_grad_(False)
        if self.verbose:
            print(f"[VJEPA21] predictor_embed: Linear({in_dim} -> {out_dim}) loaded")
        return lin

    @property
    def predictor_embed_dim(self) -> int:
        if self._predictor_embed is None:
            self._predictor_embed = self._load_predictor_embed()
        return self._predictor_embed.out_features

    @torch.no_grad()
    def compress_final(self, feat_bdhw: torch.Tensor) -> torch.Tensor:
        """Project a final-layer feature map through ``predictor_embed``.

        ``feat_bdhw``: ``(B, D, gh, gw)`` final-layer features -> ``(B, D', gh, gw)`` where
        ``D' = predictor_embed_dim`` (e.g. 768 -> 384). This is the predictor's *compressed
        input space* for the current frame (NOT a forward-predicted state).
        """
        if self._predictor_embed is None:
            self._predictor_embed = self._load_predictor_embed()
        lin = self._predictor_embed
        x = feat_bdhw.to(self.device, lin.weight.dtype)
        x = x.permute(0, 2, 3, 1)              # (B, gh, gw, D)
        x = lin(x)                             # (B, gh, gw, D')
        return x.permute(0, 3, 1, 2).contiguous()  # (B, D', gh, gw)

    def encode_images_hierarchical(
        self, images: Union[ImageLike, Sequence[ImageLike]], batch_size: int = 8
    ) -> torch.Tensor:
        """Encode frames *independently* (image mode, T=1), tapping 4 layers.

        Returns ``(N, L, grid_h * grid_w, embed_dim)`` on cpu, where ``L`` is the number
        of hierarchical layers ([2,5,8,11] for ViT-B). Each layer is the token grid for
        that frame at that depth, normed by the encoder's per-layer norm. The last layer
        (11) matches what ``encode_images`` returns.
        """
        if isinstance(images, (str, Path, Image.Image, torch.Tensor)):
            images = [images]
        self._enable_hierarchical()
        chunks = []
        for i in range(0, len(images), batch_size):
            batch = self._stack_frames(images[i : i + batch_size])     # (B,C,H,W)
            batch = batch.unsqueeze(2)                                 # (B,C,1,H,W)
            chunks.append(self._run_encoder_multi(batch))             # (B,L,N,D)
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Token geometry
    # ------------------------------------------------------------------
    def layout(self, num_frames: int = 1) -> TokenLayout:
        grid_t = 1 if num_frames == 1 else num_frames // self.tubelet_size
        return TokenLayout(
            num_frames=num_frames,
            tubelet_size=self.tubelet_size,
            grid_t=grid_t,
            grid_h=self.img_height // self.patch_size,
            grid_w=self.img_width // self.patch_size,
            patch_size=self.patch_size,
            embed_dim=self.size.embed_dim,
        )

    def unflatten(self, tokens: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        """(B, N, D) -> (B, T', grid_h, grid_w, D)."""
        lay = self.layout(num_frames)
        B, N, D = tokens.shape
        if N != lay.num_tokens:
            raise ValueError(f"Expected {lay.num_tokens} tokens for {num_frames} frames, got {N}")
        return tokens.reshape(B, lay.grid_t, lay.grid_h, lay.grid_w, D)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    def metadata(self) -> dict:
        """Everything you'd want to torch.save alongside features, once."""
        return {
            "model_size": self.size.name,
            "arch": self.size.arch,
            "embed_dim": self.size.embed_dim,
            "checkpoint": str(self.checkpoint_path),
            "encoder_key": self.encoder_key,
            "image_size": (self.img_height, self.img_width),
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "normalization": {"mean": self.IMAGENET_MEAN, "std": self.IMAGENET_STD},
            "compute_dtype": str(self.compute_dtype),
            "output_dtype": str(self.output_dtype),
        }

    def __repr__(self) -> str:
        return (
            f"VJEPA21Wrapper(size={self.size.name}, device={self.device}, "
            f"img=({self.img_height}x{self.img_width}), "
        )

