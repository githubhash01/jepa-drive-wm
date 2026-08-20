"""
Shared plumbing for the three test-set evaluators (depth, semantics, world model).

- one DEVICE, one place that knows how each checkpoint is structured and how to
  rebuild the model it belongs to (the depth decoder in particular must be built
  with the bins/norm strategies recorded in its checkpoint config);
- metric dumps: every evaluator writes a machine-readable metrics.json next to
  its figures plus a metrics.md that can be pasted into notes;
- one chart style for the summary plots, so the depth / semantics / world-model
  figures read as one set (categorical colours assigned in fixed slot order).

Nothing here knows about any particular metric -- that stays in the evaluators.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import torch

from jepa_drive_wm.models.dense_decoders.depth_decoder import DepthDecoder
from jepa_drive_wm.models.dense_decoders.semantic_decoder import SemanticDecoder
from jepa_drive_wm.models.predictors.ac_style.ac_predictor import VJEPA21WorldModel
from jepa_drive_wm.paths import OUTPUTS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default checkpoint locations (what the trainers write). Individual evaluators
# accept --checkpoint overrides; these are only the fallbacks.
DEPTH_CHECKPOINT = OUTPUTS_DIR / "checkpoints_depth" / "depth_decoder_best.pt"
SEMANTICS_CHECKPOINT = OUTPUTS_DIR / "checkpoints_semantics" / "semantic_decoder_best.pt"
WM_CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints_wm"


# ----------------------------------------------------------------------------- checkpoints

def load_checkpoint(path: Path) -> dict:
    """torch.load with the arguments every checkpoint in this repo needs."""
    # weights_only=False: the checkpoints carry plain dict/tuple metadata
    # (config, validation_metrics) alongside the state dict.
    return torch.load(path, map_location=DEVICE, weights_only=False)


def load_depth_decoder(path: Path = DEPTH_CHECKPOINT) -> tuple[DepthDecoder, dict]:
    """Rebuild the depth decoder with the readout strategies it was trained under."""
    checkpoint = load_checkpoint(path)
    config = checkpoint.get("config", {})
    model = DepthDecoder(
        bins_strategy=config.get("bins_strategy", "linear"),
        norm_strategy=config.get("norm_strategy", "linear"),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_semantic_decoder(path: Path = SEMANTICS_CHECKPOINT) -> tuple[SemanticDecoder, dict]:
    checkpoint = load_checkpoint(path)
    model = SemanticDecoder().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_world_model(path: Path) -> tuple[VJEPA21WorldModel, dict]:
    """Rebuild an ac-style world model; the checkpoint carries its action contract."""
    checkpoint = load_checkpoint(path)
    model = VJEPA21WorldModel().to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def describe_checkpoint(path: Path, checkpoint: dict) -> str:
    """One line for the console / the metrics dump: path, iteration, selection metric."""
    parts = [str(path), f"iter {checkpoint.get('iteration', '?')}"]
    if "val_ar" in checkpoint:
        parts.append(f"val ar {checkpoint['val_ar']:.4f}")
    if "step_seconds" in checkpoint:
        parts.append(f"step {checkpoint['step_seconds']:g}s")
    validation = checkpoint.get("validation_metrics", {})
    for key in ("total", "loss", "absrel", "accuracy"):
        if key in validation:
            parts.append(f"val {key} {validation[key]:.4f}")
    return " | ".join(parts)


# ----------------------------------------------------------------------------- metric dumps

def _jsonable(value: Any) -> Any:
    """Recursively coerce numpy / torch scalars and NaN into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_metrics_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(metrics), indent=2) + "\n")


def fmt(value: Any, digits: int = 4) -> str:
    """Number -> fixed-width string; NaN/None -> 'n/a' (absent class, empty scope)."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:d}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], digits: int = 4) -> str:
    """GitHub-flavoured table; the first column is left-aligned, the rest right-aligned."""
    cells = [[fmt(v, digits) for v in row] for row in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h))
              for i, h in enumerate(headers)]
    line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|" + "|".join(("-" * (w + 2)) if i == 0 else ("-" * (w + 1) + ":")
                         for i, w in enumerate(widths)) + "|"
    body = ["| " + " | ".join((c.ljust(w) if i == 0 else c.rjust(w))
                              for i, (c, w) in enumerate(zip(row, widths))) + " |"
            for row in cells]
    return "\n".join([line, sep, *body])


# ----------------------------------------------------------------------------- chart style
#
# Summary-plot styling (line / bar charts). Qualitative figures (imshow grids)
# keep their own conventions. Categorical colours are assigned by fixed slot,
# never cycled: slot 0 is always the first model / first series, and so on.

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQUENTIAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def style_axes(ax: plt.Axes, title: str | None = None, xlabel: str | None = None,
               ylabel: str | None = None) -> None:
    """Recessive chrome: hairline y-grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9.5)


def new_figure(ncols: int = 1, nrows: int = 1, width: float = 6.4, height: float = 4.2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows),
                             squeeze=False, facecolor=SURFACE)
    return fig, axes


def shared_legend_below(fig, ax: plt.Axes, ncol: int = 4) -> float:
    """
    One legend for the whole figure, under the axes, built from `ax`'s handles
    (all panels of a comparison plot share the same series). Returns the
    bottom margin (figure fraction) to reserve for it in finish_figure(rect=...).
    """
    handles, labels = ax.get_legend_handles_labels()
    rows = max(1, -(-len(labels) // ncol))
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8,
               columnspacing=1.4, handlelength=2.6)
    # ~0.2 in per legend row + padding, as a fraction of this figure's height
    return (0.2 * rows + 0.25) / fig.get_size_inches()[1]


def finish_figure(fig, path: Path, dpi: int = 150, rect: tuple | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
