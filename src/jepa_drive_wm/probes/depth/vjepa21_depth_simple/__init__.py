"""V-JEPA 2.1 linear depth probe: dense depth from frozen final-layer features."""
from .config import DepthProbeConfig
from .dataset import CachedDepthDataset, load_depth_metres, depth_collate
from .embeddings import DataPoint, ImageEmbedding, load_datapoint, load_embedding
from .head import DepthProbe, FeaturesToDepth, LinearDepthHead

__all__ = [
    "DepthProbeConfig",
    "CachedDepthDataset",
    "load_depth_metres",
    "depth_collate",
    "DataPoint",
    "ImageEmbedding",
    "load_datapoint",
    "load_embedding",
    "DepthProbe",
    "FeaturesToDepth",
    "LinearDepthHead",
]