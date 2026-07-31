import importlib
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms import v2

from jepa_drive_wm.paths import DINOV3_CKPT, DINOV3_REPO


def _load_dinov3_backbone(
    repo_dir: str | Path,
    checkpoint: str | Path,
    model_name: str,
) -> torch.nn.Module:
    # Import the builder straight from the vendored package instead of
    # torch.hub.load: hubconf.py drags in the segmentation/detection heads,
    # which need extra dependencies (torchmetrics, ...) we don't use.
    repo_dir = str(Path(repo_dir).expanduser().resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    backbones = importlib.import_module("dinov3.hub.backbones")
    model = getattr(backbones, model_name)(pretrained=False)

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


class DINOv3Encoder:
    def __init__(
        self,
        repo_dir: str | Path = DINOV3_REPO,
        checkpoint: str | Path = DINOV3_CKPT,
        model_name: str = "dinov3_vitb16",
        image_size: tuple[int, int] = (384, 1248),
        device: str | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = image_size
        self.output_dtype = output_dtype

        self.model = (
            _load_dinov3_backbone(repo_dir, checkpoint, model_name)
            .to(self.device)
            .eval()
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        patch_size = self.model.patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = tuple(patch_size)

        self.embed_dim = self.model.embed_dim
        self.num_register_tokens = self.model.n_storage_tokens

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def prepare_image(self, image: str | Path | Image.Image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            with Image.open(image) as pil_image:
                return self.transform(pil_image.convert("RGB"))

        if isinstance(image, Image.Image):
            return self.transform(image.convert("RGB"))

        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def grid_shape(self, pixel_values: torch.Tensor) -> tuple[int, int]:
        """Derive the patch-token grid from an input [B,C,H,W] tensor."""
        height, width = pixel_values.shape[-2:]
        patch_h, patch_w = self.patch_size

        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by "
                f"patch size {(patch_h, patch_w)}."
            )

        return height // patch_h, width // patch_w

    @torch.inference_mode()
    def encode_batch(
        self,
        pixel_values: torch.Tensor,
        *,
        return_grid: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: preprocessed [B,C,H,W].
            return_grid:
                False -> [B,N,D]
                True  -> [B,grid_h,grid_w,D]
        """
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        grid_h, grid_w = self.grid_shape(pixel_values)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            features = self.model.forward_features(pixel_values)

        patch_tokens = features["x_norm_patchtokens"]

        expected_tokens = grid_h * grid_w
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens} patch tokens for grid "
                f"{grid_h}x{grid_w}, but received {patch_tokens.shape[1]}."
            )

        patch_tokens = patch_tokens.to(
            device="cpu",
            dtype=self.output_dtype,
        )

        if return_grid:
            return patch_tokens.reshape(
                patch_tokens.shape[0],
                grid_h,
                grid_w,
                patch_tokens.shape[-1],
            )

        return patch_tokens

    def encode_images(
        self,
        images: str | Path | Image.Image | Sequence[str | Path | Image.Image],
        batch_size: int = 8,
        *,
        return_grid: bool = False,
    ) -> torch.Tensor:
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        else:
            images = list(images)

        outputs = []

        for start in range(0, len(images), batch_size):
            batch = torch.stack(
                [
                    self.prepare_image(image)
                    for image in images[start : start + batch_size]
                ]
            )
            outputs.append(
                self.encode_batch(batch, return_grid=return_grid)
            )

        return torch.cat(outputs, dim=0)


if __name__ == "__main__":
    from jepa_drive_wm.data.kitti import KITTISequence

    encoder = DINOv3Encoder(image_size=(384, 1248))

    seq_10 = KITTISequence(10)
    img = seq_10.left_images[0]
    tokens = encoder.encode_images(img)
    print(tokens.shape)
    # [1, 1872, 768]

    grid = encoder.encode_images(img, return_grid=True)
    print(grid.shape)
    # [1, 24, 78, 768]
