"""
Loading the cached V-JEPA 2.1 image embeddings.

The embeddings on disk were pickled by a separate encoding script, so the class
references baked into the pickle point at modules that do not exist in this repo:

    DataPoint        -> '__main__'        (the encoder script, run directly)
    ImageEmbedding   -> 'vjepa_encoder'   (a module that is not on our path)

A plain ``torch.load(..., weights_only=False)`` therefore raises
``ModuleNotFoundError``. We fix that with a small renaming unpickler that maps any
``DataPoint`` / ``ImageEmbedding`` global onto the local dataclasses below,
ignoring the original module path entirely.

The local dataclass fields mirror the originals (see data/global_pca.py).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import torch


@dataclass
class ImageEmbedding:
    image_path: str                                # ORIGINAL path -- often stale, do not trust
    original_image_dimensions: tuple[int, int]     # (height, width) of the source image
    processed_image_dimensions: tuple[int, int]    # (height, width) the encoder actually saw
    grid_h: int
    grid_w: int
    patch_size: int
    features: torch.Tensor                         # (grid_h * grid_w, feature_dim), bfloat16
    model_name: str
    checkpoint_path: str
    encoder_key: str = "ema_encoder"


@dataclass
class DataPoint:
    sequence_nr: int
    model_name: str
    timestamp: float
    frame_idx: int
    image_embedding: ImageEmbedding


# Map the pickled class *names* onto our local classes, whatever module they claim.
_LOCAL_CLASSES = {"DataPoint": DataPoint, "ImageEmbedding": ImageEmbedding}


class _RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name in _LOCAL_CLASSES:
            return _LOCAL_CLASSES[name]
        return super().find_class(module, name)


class _RenamingPickleModule:
    """A stand-in ``pickle_module`` for ``torch.load`` that uses our unpickler."""

    Unpickler = _RenamingUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _RenamingUnpickler(file).load()


def load_datapoint(path: str) -> DataPoint:
    """Load one cached ``.pt`` as a local ``DataPoint`` (handles the module-path niggle)."""
    return torch.load(path, weights_only=False, pickle_module=_RenamingPickleModule)


def load_embedding(path: str) -> ImageEmbedding:
    """Convenience: load a ``.pt`` and return just its ``ImageEmbedding``."""
    return load_datapoint(path).image_embedding
