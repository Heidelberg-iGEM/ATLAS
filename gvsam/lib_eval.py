"""
Model loading, box prediction and matching.

Shared by 02_eval_cellfinder.py, visualize.py and the notebook, so the gallery
shows exactly what the evaluation counted.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from cellSAM.sam_inference import CellSAM
from lib_io import to_uint8_rgb


def load_model(base_ckpt: str, device: str, head_ckpt: str | None = None) -> CellSAM:
    """Base CellSAM, optionally with a fine-tuned CellFinder head on top."""
    model = CellSAM({}).to(device).eval()
    model.load_state_dict(torch.load(os.path.expanduser(base_ckpt), map_location="cpu"),
                          strict=False)
    if head_ckpt:
        sd = torch.load(os.path.expanduser(head_ckpt), map_location="cpu")
        sd = sd.get("state_dict", sd)
        # a checkpoint saved from the full model carries a prefix
        for prefix in ("cellfinder.decode_head.", "module.cellfinder.decode_head."):
            if any(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
                break
        model.cellfinder.decode_head.load_state_dict(sd, strict=False)
    return model


@torch.no_grad()
def predict_boxes(model: CellSAM, img: np.ndarray, score_thr: float):
    """Boxes in pixel coordinates of the original image, plus their scores."""
    h, w = img.shape
    x = model.sam_bbox_preprocessing([to_uint8_rgb(img)], percentile=False)
    res = model.cellfinder.forward_inference(x)[0]
    boxes = res["boxes"].cpu().numpy().astype(np.float32)
    scores = res["scores"].cpu().numpy().astype(np.float32)

    keep = scores >= score_thr
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        # forward_inference works in a 1024 x 1024 frame
        boxes[:, [0, 2]] *= w / 1024.0
        boxes[:, [1, 3]] *= h / 1024.0
    return boxes, scores


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    px1, py1, px2, py2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    iw = np.maximum(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0.0)
    ih = np.maximum(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0.0)
    inter = iw * ih
    area_p = np.maximum(px2 - px1, 0.0) * np.maximum(py2 - py1, 0.0)
    area_g = np.maximum(gx2 - gx1, 0.0) * np.maximum(gy2 - gy1, 0.0)
    return (inter / np.maximum(area_p + area_g - inter, 1e-9)).astype(np.float32)


def center_size_hits(pred: np.ndarray, gt: np.ndarray, center_tol: float,
                     area_ratio: float) -> np.ndarray:
    """Does predicted box i accept annotation j?

    The centre of the annotation has to lie inside the prediction, within
    center_tol pixels, and the two areas may differ by at most area_ratio.
    """
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=bool)
    cx = ((gt[:, 0] + gt[:, 2]) / 2.0)[None, :]
    cy = ((gt[:, 1] + gt[:, 3]) / 2.0)[None, :]
    inside = ((cx >= pred[:, 0:1] - center_tol) & (cx <= pred[:, 2:3] + center_tol)
              & (cy >= pred[:, 1:2] - center_tol) & (cy <= pred[:, 3:4] + center_tol))
    area_p = np.maximum((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]), 1e-9)[:, None]
    area_g = np.maximum((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]), 1e-9)[None, :]
    return inside & (np.maximum(area_p / area_g, area_g / area_p) <= area_ratio)


def match(pred, gt, scores, mode="center", iou_thr=0.3, center_tol=2.0, area_ratio=4.0):
    """Greedy one-to-one matching, highest-scoring prediction first.

    Returns (tp, fp, fn, quality, pred_is_tp, gt_is_found), where the last two
    are boolean arrays used by the gallery to colour the boxes.
    """
    if mode == "iou":
        quality = iou_matrix(pred, gt)
        ok = quality >= iou_thr
    else:
        ok = center_size_hits(pred, gt, center_tol, area_ratio)
        quality = ok.astype(np.float32)

    pred_is_tp = np.zeros(len(pred), dtype=bool)
    gt_is_found = np.zeros(len(gt), dtype=bool)
    matched = []

    for i in (np.argsort(-scores) if len(scores) else []):
        free = [j for j in np.where(ok[i])[0] if not gt_is_found[j]]
        if not free:
            continue
        j = max(free, key=lambda j: quality[i, j])
        pred_is_tp[i] = True
        gt_is_found[j] = True
        matched.append(float(quality[i, j]))

    tp = int(pred_is_tp.sum())
    return tp, len(pred) - tp, len(gt) - tp, matched, pred_is_tp, gt_is_found
