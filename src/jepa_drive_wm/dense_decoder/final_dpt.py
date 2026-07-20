"""
Model: 


[VJEPA2.1, VJEPA2.1, VJEPA2.1, VJEPA2.1] -> DINOV3 DPT Head -> Dense Predictions (Depth, Semantic Segmentation, Surface Normals, etc.)


"""
from jepa_drive_wm.dense_decoder.dinov3_files.dinov3_dpt import DPTHead
import torch
from torch import nn


"""
Generic Final Feature DPT Head for VJEPA2.1-BASE, using DINOV3 DPT Head 
"""
class FinalFeatureDPT(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        output_channels: int = 256): 
        super().__init__()
        
        self.dpt = DPTHead(
            in_channels=(embed_dim,) * 4,
            channels=256,
            post_process_channels=[128, 256, 512, 1024],
            readout_type="ignore",
            expand_channels=False,
            n_output_channels=output_channels,
            n_hidden_channels=32,
            use_batchnorm=False,
            use_bias=False,
            projection_after_fusion=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(
                f"Expected V-JEPA grid [B, C, H, W], got {tuple(z.shape)}"
            )

        # DINOv3's ReassembleBlocks expects (patch_map, cls_token) pairs
        # even when readout_type='ignore'. The dummy token is never used.
        batch_size, channels, _, _ = z.shape
        unused_cls_token = z.new_zeros(batch_size, channels)

        dpt_inputs = [
            (z, unused_cls_token)
            for _ in range(4)
        ]
        return self.dpt(dpt_inputs)
    

