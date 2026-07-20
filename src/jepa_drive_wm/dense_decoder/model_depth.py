import torch
from torch import nn
from jepa_drive_wm.dense_decoder.dinov3_files.depth_utils import FeaturesToDepth
from jepa_drive_wm.dense_decoder.final_dpt import FinalFeatureDPT

# DepthHead
class DepthDecoder(nn.Module):

    def __init__(self, embed_dim: int = 768):  
        super().__init__()
        
        # Repeated DPTHead for depth prediction, followed by FeaturesToDepth for depth binning and normalization
        self.depth_model = nn.Sequential(
            FinalFeatureDPT(
                embed_dim=embed_dim,
                output_channels=256,
                use_input_batchnorm=False,
            ),
            FeaturesToDepth(
                min_depth=0.5,
                max_depth=80.0
            ),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(
                f"Expected V-JEPA grid [B, C, H, W], got {tuple(z.shape)}"
            )
        return self.depth_model(z)
