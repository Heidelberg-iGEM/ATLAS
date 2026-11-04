#!/usr/bin/env python3
"""
Stage 1: turn hand-drawn instance masks into a bounding box table.

    python annotation_utils.py --image_dir 01_raw --mask_dir 02_masks_gv \\
        --out_csv 03_splits/all_bboxes.tsv

A mask has background 0 and one positive integer id per annotated cluster,
which is what napari's Labels layer writes. Every id becomes the smallest
rectangle around it. Those boxes are the training targets for CellFinder and
the prompts for the mask decoder.

Columns: uid, image_path, label_path, xmin, ymin, xmax, ymax, label_id

An image whose mask has no instances is kept as a single row with empty box
columns. That keeps it in the table as an image with zero targets, which is how
negative examples reach the detector. Whether an empty mask means "checked,
nothing there" or "not annotated yet" is decided in stage 2.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops


def label_mask_to_rows(label_img: np.ndarray, uid: str, image_path: str,
                       label_path: str, keep_empty: bool = False) -> pd.DataFrame:
    rows = []
    for region in regionprops(label_img):
        y0, x0, y1, x1 = region.bbox     # skimage: (min_row, min_col, max_row, max_col)
        rows.append({"uid": uid, "image_path": image_path, "label_path": label_path,
                     "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
                     "label_id": region.label})
    if not rows and keep_empty:
        rows.append({"uid": uid, "image_path": image_path, "label_path": label_path,
                     "xmin": np.nan, "ymin": np.nan, "xmax": np.nan, "ymax": np.nan,
                     "label_id": np.nan})
    return pd.DataFrame(rows)


def build_bboxes_csv(image_dir: str, mask_dir: str, out_csv: str,
                     image_ext: str = ".tif", mask_suffix: str = "_mask",
                     keep_empty: bool = True) -> pd.DataFrame:
    frames, n_empty = [], 0
    for image_path in sorted(glob.glob(os.path.join(image_dir, f"*{image_ext}"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(mask_dir, f"{stem}{mask_suffix}{image_ext}")
        if not os.path.exists(label_path):
            print(f"  no mask for {stem}, skipped")
            continue
        df = label_mask_to_rows(tiff.imread(label_path), stem, image_path,
                                label_path, keep_empty=keep_empty)
        if df.empty:
            continue
        n_empty += int(df["xmin"].isna().all())
        frames.append(df)

    if not frames:
        raise ValueError(f"no matching image and mask pairs in {image_dir} and {mask_dir}")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_csv, sep="\t", index=False)
    n_boxes = int(out["xmin"].notna().sum())
    print(f"  {n_boxes} boxes across {out['uid'].nunique()} images "
          f"({n_empty} without annotation) -> {out_csv}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--image_ext", default=".tif")
    ap.add_argument("--mask_suffix", default="_mask",
                    help="mask filename = <image stem><suffix><ext>")
    a = ap.parse_args()
    build_bboxes_csv(a.image_dir, a.mask_dir, a.out_csv, a.image_ext, a.mask_suffix)
