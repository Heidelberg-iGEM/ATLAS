"""
Shared image IO for the GVSAM stages.

These four functions were duplicated across the training and evaluation
scripts. Keeping them in one place means a change to the CZI handling or the
normalisation cannot silently affect training and evaluation differently.
"""
from __future__ import annotations

import os

import numpy as np
import tifffile as tiff
import torch


def read_image_any(path: str) -> np.ndarray:
    """Return a 2D array (Y, X).

    TIFF is read as is. A 3D stack is max-projected over the first axis, which
    is what our microscope writes for z-stacks. CZI files are read through
    bioio and projected the same way.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        arr = np.squeeze(tiff.imread(path))
        if arr.ndim == 3:
            arr = arr.max(axis=0)
        if arr.ndim != 2:
            raise ValueError(f"{path}: not 2D after squeeze and projection, shape={arr.shape}")
        return arr.astype(np.float32, copy=False)

    if ext == ".czi":
        from bioio import BioImage
        d = np.asarray(BioImage(path).data)
        if d.ndim == 5:        # T C Z Y X
            zyx = d[0, 0]
        elif d.ndim == 4:      # C Z Y X or T Z Y X
            zyx = d[0]
        elif d.ndim == 3:      # Z Y X
            zyx = d
        else:
            raise ValueError(f"{path}: unsupported CZI shape {d.shape}")
        return zyx.max(axis=0).astype(np.float32, copy=False)

    raise ValueError(f"{path}: unsupported extension {ext}")


def to_uint8_rgb(img2d: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> torch.Tensor:
    """Contrast-stretch between two percentiles and repeat to three channels.

    Percentiles rather than min and max, because a single hot pixel would
    otherwise compress the whole image into a narrow range. Returns (3, H, W)
    uint8, which is what CellSAM's bbox preprocessing expects.
    """
    x = img2d.astype(np.float32)
    lo, hi = np.percentile(x, p_low), np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    u8 = (x * 255.0).astype(np.uint8)
    return torch.from_numpy(np.stack([u8, u8, u8], axis=0))


def to_cellsam_float(img2d: np.ndarray) -> torch.Tensor:
    """(3, H, W) float32 using CellSAM's own normalisation.

    The decoder path needs this instead of to_uint8_rgb, because the mask
    decoder was trained on CellSAM-normalised input.
    """
    from cellSAM.utils import format_image_shape, normalize_image
    img = normalize_image(format_image_shape(img2d)).astype(np.float32)
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))


def boxes_to_cxcywh_norm(g, width: int, height: int) -> np.ndarray:
    """Pixel xyxy boxes from the TSV to normalised cxcywh, as DETR expects."""
    cx = ((g["xmin"] + g["xmax"]) / 2.0).to_numpy(np.float32) / width
    cy = ((g["ymin"] + g["ymax"]) / 2.0).to_numpy(np.float32) / height
    w = (g["xmax"] - g["xmin"]).to_numpy(np.float32) / width
    h = (g["ymax"] - g["ymin"]).to_numpy(np.float32) / height
    return np.clip(np.stack([cx, cy, w, h], axis=1).astype(np.float32), 0.0, 1.0)


def inside_border(boxes_xyxy: np.ndarray, height: int, width: int,
                  margin: int = 1) -> np.ndarray:
    """Which boxes do not touch the image border?

    A cluster cut off by the frame cannot be annotated reliably, so counting it
    either way adds noise rather than signal.

    Returns a boolean mask and not the boxes themselves, so the caller can drop
    the matching scores with it.
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros(0, dtype=bool)
    b = boxes_xyxy.astype(np.float32, copy=True)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width - 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height - 1)
    return ((b[:, 0] > margin) & (b[:, 1] > margin)
            & (b[:, 2] < width - 1 - margin) & (b[:, 3] < height - 1 - margin))
