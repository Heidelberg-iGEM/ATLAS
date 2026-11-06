"""GVSAM inference: segment gas vesicle clusters in new images with the shipped weights.

    python predict.py IMAGE_OR_FOLDER [...] --outdir preds

Writes <image>_mask.tif for every image (uint16, background 0, one id per cluster).
Uses weights/decoder.pt (the full fine-tuned CellSAM) next to this script; pass
--ckpt for a decoder.pt from your own run. Use absolute paths for the images when
started through atlas.sh, which runs the script from work/gvsam/00_pipeline.
"""
import argparse
import glob
import os

import numpy as np
import tifffile as tiff
import torch
from cellSAM.model import get_local_model

from lib_io import read_image_any, to_cellsam_float

HERE = os.path.dirname(os.path.realpath(__file__))   # resolves the symlink made by install.sh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or folders (.tif, .tiff, .czi)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "weights", "decoder.pt"))
    ap.add_argument("--bbox_threshold", type=float, default=0.4,
                    help="detector confidence; lower finds more clusters (config.yaml decoder.bbox_threshold)")
    a = ap.parse_args()

    ckpt = os.path.expanduser(a.ckpt)
    if os.path.getsize(ckpt) < 10_000_000:
        raise SystemExit(f"{ckpt} is a Git LFS pointer, not the weights. Run: git lfs pull")

    paths = []
    for p in a.inputs:
        if os.path.isdir(p):
            for ext in ("tif", "tiff", "czi"):
                paths += sorted(glob.glob(os.path.join(p, f"*.{ext}")))
        else:
            paths.append(p)
    if not paths:
        raise SystemExit("no images found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_local_model(ckpt).to(device).eval()
    model.bbox_threshold = a.bbox_threshold
    os.makedirs(a.outdir, exist_ok=True)

    for p in paths:
        img = read_image_any(p)
        with torch.no_grad():
            labels, *_ = model.predict(to_cellsam_float(img).unsqueeze(0).to(device),
                                       boxes_per_heatmap=None)
        labels = (np.zeros(img.shape) if labels is None else np.asarray(labels)).astype(np.uint16)
        out = os.path.join(a.outdir, os.path.splitext(os.path.basename(p))[0] + "_mask.tif")
        tiff.imwrite(out, labels)
        print(f"{os.path.basename(p)}: {len(np.unique(labels)) - 1} clusters -> {out}")


if __name__ == "__main__":
    main()
