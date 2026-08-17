import torch
from torch import nn
from jepa_drive_wm.models.dense_decoders.dinov3_files.depth_utils import FeaturesToDepth
from jepa_drive_wm.models.dense_decoders.final_dpt import FinalFeatureDPT

# DepthHead
class DepthDecoder(nn.Module):

    def __init__(self, embed_dim: int = 768, bins_strategy: str = "linear", norm_strategy: str = "linear"):
        super().__init__()

        # Repeated DPTHead for depth prediction, followed by FeaturesToDepth for depth binning and normalization.
        # The strategies add no parameters, so the state dict is identical across them:
        # a checkpoint must be loaded with the strategies it was trained under (saved in its config).
        self.depth_model = nn.Sequential(
            FinalFeatureDPT(
                embed_dim=embed_dim,
                output_channels=256
            ),
            FeaturesToDepth(
                min_depth=0.5,
                max_depth=80.0,
                bins_strategy=bins_strategy,
                norm_strategy=norm_strategy,
            ),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(
                f"Expected V-JEPA grid [B, C, H, W], got {tuple(z.shape)}"
            )
        return self.depth_model(z)
