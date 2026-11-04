#!/usr/bin/env python3
"""
Stage 2: split into train, validation and test, then augment the training split.

    python augment.py --image_dir 01_raw --mask_dir 02_masks_gv \\
        --out_dir 03_splits --angles 90 270 --negatives negatives.txt

The order matters. Splitting first and augmenting second keeps a rotated copy
of a training image out of validation, which would otherwise make the model
look better than it is.

Rotations by multiples of 90 degrees are lossless: no interpolation, no edge
artefacts, and the instance ids survive unchanged. Other angles go through
scipy, which crops corners on non-square images and pads the edges, so they are
not used by default.

Augmented files carry "_aug" in their uid. The training script draws them less
often through a weighted sampler, so the model does not mostly see rotations.

Empty masks are ambiguous: either the image was checked and holds no gas
vesicles, or it has not been annotated yet. Only stems listed in --negatives
are used as negative examples, the rest are dropped. Negatives are split
separately so that validation and test contain some too.

Output:
    <out_dir>/aug_images/, aug_masks/     augmented copies
    <out_dir>/train_bboxes.tsv            originals plus augmentations
    <out_dir>/val_bboxes.tsv              originals only
    <out_dir>/test_bboxes.tsv             originals only
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import rotate as ndi_rotate

from annotation_utils import label_mask_to_rows


def rotate_pair(image, mask, angle):
    """Rotate image and mask identically."""
    if float(angle) % 90 == 0:
        k = int(round(float(angle) / 90)) % 4
        return np.rot90(image, k).copy(), np.rot90(mask, k).copy()
    # order=0 for the mask, otherwise interpolation invents instance ids
    img = ndi_rotate(image, angle, reshape=False, order=3, mode="reflect").astype(image.dtype)
    msk = ndi_rotate(mask.astype(np.float32), angle, reshape=False, order=0,
                     mode="constant", cval=0).astype(mask.dtype)
    return img, msk


def flip_pair(image, mask, mode):
    if mode == "horizontal":
        return np.fliplr(image).copy(), np.fliplr(mask).copy()
    if mode == "vertical":
        return np.flipud(image).copy(), np.flipud(mask).copy()
    raise ValueError(f"unknown flip mode: {mode}")


def split_stems(stems, val_frac, test_frac, seed, min_one=True):
    """Split image stems into train, val and test.

    min_one forces at least one image into val and test as long as the fraction
    is above zero. With 3 images, round(3 * 0.1) would otherwise be 0 and the
    test set would be empty.
    """
    stems = list(stems)
    np.random.default_rng(seed).shuffle(stems)
    n = len(stems)
    if not min_one:
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        return stems[:n - n_val - n_test], stems[n - n_val - n_test:n - n_test], stems[n - n_test:]

    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
    if n - n_val - n_test <= 0:
        raise ValueError("split fractions too large, the training split would be empty")
    return stems[:n - n_val - n_test], stems[n - n_val - n_test:n - n_test], stems[n - n_test:]


def read_stem_list(path):
    stems = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                stems.add(os.path.splitext(line)[0])
    return stems


def n_instances(mask_path):
    ids = np.unique(np.asarray(tiff.imread(mask_path)))
    return int(ids[ids != 0].size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_ext", default=".tif")
    ap.add_argument("--mask_suffix", default="_mask")
    ap.add_argument("--angles", type=float, nargs="*", default=[90, 270])
    ap.add_argument("--flips", nargs="*", default=[], choices=["horizontal", "vertical"])
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negatives", default=None,
                    help="text file of image stems whose empty mask means "
                         "'checked, no gas vesicles'")
    a = ap.parse_args()

    aug_img_dir = os.path.join(a.out_dir, "aug_images")
    aug_msk_dir = os.path.join(a.out_dir, "aug_masks")
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_msk_dir, exist_ok=True)

    # Find image and mask pairs
    pairs = {}
    for image_path in sorted(glob.glob(os.path.join(a.image_dir, f"*{a.image_ext}"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(a.mask_dir, f"{stem}{a.mask_suffix}{a.image_ext}")
        if os.path.exists(mask_path):
            pairs[stem] = (image_path, mask_path)
        else:
            print(f"  no mask for {stem}, skipped")
    if not pairs:
        raise ValueError("no matching image and mask pairs found")

    # Sort out empty masks
    neg_list = read_stem_list(a.negatives) if a.negatives else set()
    annotated, negatives, unannotated = {}, {}, []
    for stem, paths in pairs.items():
        if n_instances(paths[1]) > 0:
            annotated[stem] = paths
        elif stem in neg_list:
            negatives[stem] = paths
        else:
            unannotated.append(stem)

    print(f"  {len(annotated)} annotated, {len(negatives)} negative examples")
    if unannotated:
        print(f"  {len(unannotated)} masks are empty and not listed in --negatives, "
              f"excluded: {', '.join(unannotated[:6])}"
              + (" ..." if len(unannotated) > 6 else ""))
    if len(annotated) < 3:
        raise ValueError("too few annotated images for a three-way split")

    # Split annotated and negative images separately, so each subset has both
    train, val, test = split_stems(annotated, a.val_frac, a.test_frac, a.seed)
    n_train, n_val, n_test = split_stems(negatives, a.val_frac, a.test_frac, a.seed,
                                         min_one=len(negatives) >= 3)
    print(f"  annotated: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  negative:  train={len(n_train)} val={len(n_val)} test={len(n_test)}")
    train, val, test = train + n_train, val + n_val, test + n_test
    pairs = {**annotated, **negatives}

    def original_rows(stem):
        image_path, mask_path = pairs[stem]
        return label_mask_to_rows(tiff.imread(mask_path), stem, image_path,
                                  mask_path, keep_empty=True)

    val_rows = [original_rows(s) for s in val]
    test_rows = [original_rows(s) for s in test]

    # Training split: originals plus augmentations
    train_rows = []
    for stem in train:
        image_path, mask_path = pairs[stem]
        train_rows.append(original_rows(stem))
        image, mask = tiff.imread(image_path), tiff.imread(mask_path)

        variants = [(f"rot{int(ang)}", *rotate_pair(image, mask, ang)) for ang in a.angles]
        variants += [(f"flip{m[:1]}", *flip_pair(image, mask, m)) for m in a.flips]

        for tag, img, msk in variants:
            uid = f"{stem}_aug_{tag}"
            out_img = os.path.join(aug_img_dir, f"{uid}{a.image_ext}")
            out_msk = os.path.join(aug_msk_dir, f"{uid}{a.mask_suffix}{a.image_ext}")
            tiff.imwrite(out_img, img)
            tiff.imwrite(out_msk, msk, compression="zlib")
            train_rows.append(label_mask_to_rows(msk, uid, out_img, out_msk, keep_empty=True))

    for rows, name in ((train_rows, "train"), (val_rows, "val"), (test_rows, "test")):
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        path = os.path.join(a.out_dir, f"{name}_bboxes.tsv")
        df.to_csv(path, sep="\t", index=False)
        n_img = df["uid"].nunique() if not df.empty else 0
        n_box = int(df["xmin"].notna().sum()) if not df.empty else 0
        print(f"  {name}: {n_box} boxes across {n_img} images -> {path}")


if __name__ == "__main__":
    main()
