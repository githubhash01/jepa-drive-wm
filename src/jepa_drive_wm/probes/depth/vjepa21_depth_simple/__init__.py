"""V-JEPA 2.1 linear depth probe: dense depth from frozen final-layer features."""
from .config import DepthProbeConfig
from .dataset import CachedDepthDataset, load_depth_metres, depth_collate
from .head import DepthProbe, FeaturesToDepth, LinearDepthHead, DPTDepthHead

__all__ = [
    "DepthProbeConfig",
    "CachedDepthDataset",
    "load_depth_metres",
    "depth_collate",
    "DepthProbe",
    "FeaturesToDepth",
    "LinearDepthHead",
    "DPTDepthHead",
]
