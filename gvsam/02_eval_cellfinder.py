#!/usr/bin/env python3
"""
Stage 4: evaluate the fine-tuned detector against the base model.

    python 02_eval_cellfinder.py --bboxes_test 03_splits/test_bboxes.tsv \\
        --new_ckpt 04_runs/ft1/cellfinder.pt --outdir 04_runs/ft1

Both models run over the same test images, so the numbers are comparable.

MATCHING
For every predicted box the model returns a score between 0 and 1. Boxes below
--score_thr are discarded; a lower threshold finds more clusters and produces
more false positives.

The usual way to match a prediction to an annotation is intersection over
union. For objects as small as gas vesicle clusters a shift of one pixel
already costs a lot of IoU, so the default here is a criterion on position and
size instead:

    position   the centre of the annotated box lies inside the predicted box,
               with a tolerance of --center_tol pixels
    size       the two areas differ by at most a factor of --area_ratio

Each annotation can be matched once, and higher-scoring predictions are matched
first. Use --match iou for the classical criterion.

The matching itself lives in lib_eval.py, so the gallery colours boxes by
exactly what was counted here.

    TP  predicted boxes that match an annotation
    FP  predicted boxes without one
    FN  annotated clusters that no box matched

    recall = TP / (TP + FN), precision = TP / (TP + FP), F1 the harmonic mean

Images without gas vesicles contribute only false positives, since there is
nothing to recall.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lib_eval import load_model, match, predict_boxes
from lib_io import inside_border, read_image_any


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bboxes_test", required=True)
    ap.add_argument("--new_ckpt", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score_thr", type=float, default=0.2)
    ap.add_argument("--match", default="center", choices=["center", "iou"])
    ap.add_argument("--center_tol", type=float, default=2.0)
    ap.add_argument("--area_ratio", type=float, default=4.0)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--border_margin", type=int, default=1)
    a = ap.parse_args()

    groups = list(pd.read_csv(a.bboxes_test, sep="\t").groupby("uid"))
    n_gt = sum(len(g.dropna(subset=["xmin"])) for _, g in groups)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  {len(groups)} test images, {n_gt} annotated clusters, "
          f"matching by {a.match}")

    models = {"base_cellsam": load_model(a.base_ckpt, device),
              "gvsam": load_model(a.base_ckpt, device, a.new_ckpt)}

    results, per_image = [], []
    for name, model in models.items():
        TP = FP = FN = 0
        qualities = []
        for uid, g in tqdm(groups, desc=name, leave=False):
            img = read_image_any(g["image_path"].iloc[0])
            gt = g[["xmin", "ymin", "xmax", "ymax"]].dropna().to_numpy(np.float32)
            pred, scores = predict_boxes(model, img, a.score_thr)
            keep = inside_border(pred, *img.shape, margin=a.border_margin)
            pred, scores = pred[keep], scores[keep]

            tp, fp, fn, q, _, _ = match(pred, gt, scores, mode=a.match,
                                        iou_thr=a.iou_thr, center_tol=a.center_tol,
                                        area_ratio=a.area_ratio)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            qualities.extend(q)
            per_image.append({"model": name, "uid": uid, "n_gt": len(gt),
                              "n_pred": len(pred), "TP": tp, "FP": fp, "FN": fn})

        precision = TP / max(TP + FP, 1)
        recall = TP / max(TP + FN, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        results.append({"model": name, "TP": TP, "FP": FP, "FN": FN,
                        "precision": round(precision, 4), "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "mean_quality": round(float(np.mean(qualities)), 4) if qualities else 0.0})

    df = pd.DataFrame(results)
    os.makedirs(a.outdir, exist_ok=True)
    df.to_csv(os.path.join(a.outdir, "eval_cellfinder.tsv"), sep="\t", index=False)
    pd.DataFrame(per_image).to_csv(os.path.join(a.outdir, "eval_per_image.tsv"),
                                   sep="\t", index=False)

    print()
    print(df.to_string(index=False))
    print(f"\n  -> {a.outdir}/eval_cellfinder.tsv and eval_per_image.tsv")
    print("  Recall is the share of annotated clusters found, precision the share")
    print("  of predictions that were right. Per-image numbers vary with image")
    print("  quality and background, so look at both tables.")


if __name__ == "__main__":
    main()
