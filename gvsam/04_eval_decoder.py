#!/usr/bin/env python3
"""
Stage 6: evaluate the fine-tuned mask decoder against base CellSAM. Optional.

    python 04_eval_decoder.py --split_dir 03_splits --outdir 04_runs/ft1 \\
        --base_ckpt ~/.deepcell/.../cellsam_general.pt \\
        --new_ckpt 04_runs/ft1/decoder.pt

Both models run the full pipeline on the test images, detector and decoder
together, so what is measured here is the instance masks the pipeline actually
produces.

Two levels, because they fail differently:

    pixel       every pixel counted as foreground or background. Says whether
                the masks have the right extent.
    instance    predicted and annotated clusters matched one to one by IoU,
                best pair first, above --inst_iou_thr. Says whether the right
                number of clusters was found.

A model can look good on pixels and bad on instances by merging two touching
clusters into one mask, which is the failure mode that matters for counting.

PQ is the panoptic quality of the matched instances, sum of their IoU divided by
TP + FP/2 + FN/2, so it mixes how many were found with how well they fit.

Predicted instances touching the image border are dropped, as in stage 4: a
cluster cut off by the frame cannot be annotated reliably.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from scipy.ndimage import find_objects
from tqdm import tqdm

from cellSAM.model import get_local_model
from lib_io import read_image_any, to_cellsam_float


def compact(labels: np.ndarray) -> np.ndarray:
    """Relabel instances to 1..N, keeping background at 0."""
    labels = labels.astype(np.int64, copy=False)
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros_like(labels, dtype=np.uint16)
    remap = np.zeros(int(ids.max()) + 1, dtype=np.uint32)
    remap[ids] = np.arange(1, ids.size + 1)
    return remap[labels].astype(np.uint16)


def drop_border_instances(labels: np.ndarray, margin: int = 1) -> np.ndarray:
    if margin <= 0:
        return compact(labels)
    lab = compact(labels)
    h, w = lab.shape
    keep = np.zeros(int(lab.max()) + 1, dtype=bool)
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = sl
        keep[i] = (ys.start > margin and xs.start > margin
                   and ys.stop < h - margin and xs.stop < w - margin)
    lab[~keep[lab]] = 0
    return compact(lab)


def instance_match(gt: np.ndarray, pred: np.ndarray, iou_thr: float):
    """One-to-one matching by IoU, best pair first. Returns tp, fp, fn, ious."""
    gt, pred = compact(gt), compact(pred)
    n_gt, n_pred = int(gt.max()), int(pred.max())
    if n_gt == 0 or n_pred == 0:
        return 0, n_pred, n_gt, []

    # Pair counts in one pass: the histogram of pred_id * (n_gt + 1) + gt_id
    # holds the intersection of every pair.
    k = n_gt + 1
    inter = np.bincount(pred.ravel().astype(np.int64) * k + gt.ravel().astype(np.int64),
                        minlength=(n_pred + 1) * k).reshape(n_pred + 1, k).astype(np.float64)
    area_pred = np.bincount(pred.ravel(), minlength=n_pred + 1).astype(np.float64)
    area_gt = np.bincount(gt.ravel(), minlength=k).astype(np.float64)

    inter = inter[1:, 1:]
    union = area_pred[1:, None] + area_gt[None, 1:] - inter
    iou = inter / np.maximum(union, 1e-9)

    pairs = [(float(iou[p, g]), p, g) for p, g in zip(*np.where(iou >= iou_thr))]
    pairs.sort(reverse=True)

    used_p, used_g, matched = set(), set(), []
    for value, p, g in pairs:
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        matched.append(value)

    tp = len(matched)
    return tp, n_pred - tp, n_gt - tp, matched


def prf(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-12)


@torch.no_grad()
def predict_labels(model, img, bbox_threshold):
    model.bbox_threshold = float(bbox_threshold)
    batch = to_cellsam_float(img).unsqueeze(0).to(next(model.parameters()).device)
    labels, *_ = model.predict(batch, boxes_per_heatmap=None)
    if labels is None:
        return np.zeros(img.shape, dtype=np.uint16)
    return np.asarray(labels).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_dir", required=True, help="folder with test_bboxes.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--new_ckpt", required=True, help="decoder.pt from stage 5")
    ap.add_argument("--bbox_threshold", type=float, default=0.4,
                    help="detector threshold, the same for both models")
    ap.add_argument("--inst_iou_thr", type=float, default=0.5)
    ap.add_argument("--border_margin", type=int, default=1)
    ap.add_argument("--save_preds", action="store_true",
                    help="write the predicted instance masks as TIFF")
    a = ap.parse_args()

    groups = list(pd.read_csv(os.path.join(a.split_dir, "test_bboxes.tsv"), sep="\t")
                  .dropna(subset=["label_path"]).groupby("uid"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  {len(groups)} test images, instance matching at IoU "
          f"{a.inst_iou_thr}, device {device}")

    models = {"base_cellsam": get_local_model(os.path.expanduser(a.base_ckpt)),
              "gvsam": get_local_model(os.path.expanduser(a.new_ckpt))}
    for m in models.values():
        m.to(device).eval()

    pred_dir = os.path.join(a.outdir, "test_preds")
    if a.save_preds:
        os.makedirs(pred_dir, exist_ok=True)

    results, per_image = [], []
    for name, model in models.items():
        px = np.zeros(3, dtype=np.int64)        # TP, FP, FN over all pixels
        inst = np.zeros(3, dtype=np.int64)
        ious = []

        for uid, g in tqdm(groups, desc=name, leave=False):
            img = read_image_any(g["image_path"].iloc[0])
            gt = np.asarray(tiff.imread(g["label_path"].iloc[0])).astype(np.uint16)
            pred = drop_border_instances(predict_labels(model, img, a.bbox_threshold),
                                         a.border_margin)

            gt_fg, pred_fg = gt > 0, pred > 0
            px += [np.logical_and(pred_fg, gt_fg).sum(),
                   np.logical_and(pred_fg, ~gt_fg).sum(),
                   np.logical_and(~pred_fg, gt_fg).sum()]

            tp, fp, fn, matched = instance_match(gt, pred, a.inst_iou_thr)
            inst += [tp, fp, fn]
            ious.extend(matched)
            per_image.append({"model": name, "uid": uid, "n_gt": int(compact(gt).max()),
                              "n_pred": int(pred.max()), "TP": tp, "FP": fp, "FN": fn})

            if a.save_preds and name == "gvsam":
                tiff.imwrite(os.path.join(pred_dir, f"{uid}_pred_labels.tif"),
                             pred, compression="zlib")

        px_p, px_r, px_f1 = prf(*px)
        in_p, in_r, in_f1 = prf(*inst)
        denom = inst[0] + 0.5 * inst[1] + 0.5 * inst[2]
        results.append({
            "model": name,
            "px_precision": round(px_p, 4), "px_recall": round(px_r, 4),
            "px_f1": round(px_f1, 4),
            "inst_TP": int(inst[0]), "inst_FP": int(inst[1]), "inst_FN": int(inst[2]),
            "inst_precision": round(in_p, 4), "inst_recall": round(in_r, 4),
            "inst_f1": round(in_f1, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "PQ": round(float(np.sum(ious)) / denom, 4) if denom > 0 else 0.0,
        })

    df = pd.DataFrame(results)
    os.makedirs(a.outdir, exist_ok=True)
    df.to_csv(os.path.join(a.outdir, "eval_decoder.tsv"), sep="\t", index=False)
    pd.DataFrame(per_image).to_csv(os.path.join(a.outdir, "eval_decoder_per_image.tsv"),
                                   sep="\t", index=False)

    print()
    print(df.to_string(index=False))
    print(f"\n  -> {a.outdir}/eval_decoder.tsv and eval_decoder_per_image.tsv")
    print("  Pixel metrics well above the instance metrics mean the masks fit but")
    print("  clusters are being merged or split. Compare n_gt and n_pred per image.")


if __name__ == "__main__":
    main()
