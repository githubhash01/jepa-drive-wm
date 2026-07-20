import torch
from torch import nn
from jepa_drive_wm.dense_decoder.final_dpt import FinalFeatureDPT

# Semantics Head
class SemanticDecoder(nn.Module):

    def __init__(self, embed_dim: int = 768):  
        super().__init__()
        
        self.semantic_model = nn.Sequential(
            FinalFeatureDPT(
                embed_dim=embed_dim,
                output_channels=256,
                use_input_batchnorm=False,
            )
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(
                f"Expected V-JEPA grid [B, C, H, W], got {tuple(z.shape)}"
            )
        return self.semantic_model(z)
