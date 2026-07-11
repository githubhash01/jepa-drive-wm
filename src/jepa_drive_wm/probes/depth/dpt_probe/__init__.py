"""4-layer DPT depth probe: dense depth by fusing intermediate V-JEPA 2.1 layers ([2,5,8,11])."""
from .config import DPTProbeConfig
from .dataset import ImageDepthDataset
from .head import DPTDepthHead, DptDepthProbe

__all__ = ["DPTProbeConfig", "ImageDepthDataset", "DptDepthProbe", "DPTDepthHead"]
