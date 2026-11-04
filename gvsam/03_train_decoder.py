#!/usr/bin/env python3
"""
Stage 5: fine-tune SAM's mask decoder on gas vesicle clusters. Optional.

    python 03_train_decoder.py --split_dir 03_splits --outdir 04_runs/ft1 \\
        --init_cellsam_ckpt ~/.deepcell/.../cellsam_general.pt \\
        --cellfinder_ckpt 04_runs/ft1/cellfinder.pt

Only needed if pixel masks are wanted instead of boxes, for example to measure
cluster area. For counting clusters per cell the boxes from stage 3 are enough,
which is why decoder.enabled is false by default.

The decoder turns a box into a mask. During training it is prompted with the
annotated boxes, not with CellFinder's predictions, so a weak detector cannot
spoil the mask training. At inference CellFinder provides the boxes, which is
why the detector has to be good first.

Trained: the mask decoder and the encoder's neck. The ViT encoder and CellFinder
stay frozen. Loss is BCE plus Dice per instance, averaged over all instances in
the batch, so images with many clusters do not dominate.

Writes decoder.pt, the full CellSAM state dict with the fine-tuned decoder in
it, plus decoder_losses.tsv and decoder_loss_curve.png.
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from cellSAM.sam_inference import CellSAM
from lib_io import read_image_any, to_cellsam_float

BOX_COLS = ["xmin", "ymin", "xmax", "ymax", "label_id"]


class MaskDataset(Dataset):
    """One item per image: the image, its instance mask, and the boxes.

    Rows without a box are dropped. The decoder learns masks for boxes, so an
    image without one has nothing to contribute.
    """

    def __init__(self, tsv, max_instances, seed=0):
        df = pd.read_csv(tsv, sep="\t").dropna(subset=BOX_COLS)
        self.groups = list(df.groupby("uid"))
        self.max_instances = int(max_instances)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        uid, g = self.groups[idx]
        rows = g[BOX_COLS]
        if 0 < self.max_instances < len(rows):
            # cap the instances per image, otherwise one crowded image costs as
            # much GPU memory as a whole batch
            take = self.rng.choice(len(rows), self.max_instances, replace=False)
            rows = rows.iloc[np.sort(take)]
        return {
            "uid": str(uid),
            "img": to_cellsam_float(read_image_any(g["image_path"].iloc[0])),
            "label_img": np.asarray(tiff.imread(g["label_path"].iloc[0])),
            "boxes": rows[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32),
            "label_ids": rows["label_id"].to_numpy(int),
        }


def collate(batch):
    return batch


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + eps) / (prob.sum() + target.sum() + eps)


def masks_for_boxes(cellsam, boxes_xyxy, size_hw, padding, embedding):
    """Mask logits for one image, one per box, at the original image size."""
    h, w = size_hw
    mdl = cellsam.model_cp
    device = embedding.device

    # The prompt encoder works in the padded 1024 x 1024 frame
    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, [0, 2]] *= 1024.0 / w
    boxes[:, [1, 3]] *= 1024.0 / h
    input_size = torch.tensor([1024 - padding[0], 1024 - padding[1]], device=device)

    out = []
    for b in boxes:
        box = torch.as_tensor(b, dtype=torch.float32, device=device)[None, None, :]
        sparse, dense = mdl.prompt_encoder(points=None, boxes=box, masks=None)
        low_res, _ = mdl.mask_decoder(
            image_embeddings=embedding.unsqueeze(0),
            image_pe=mdl.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        up = mdl.postprocess_masks(low_res, input_size=input_size, original_size=(h, w))
        out.append(up[0, 0])
    return out


def batch_loss(cellsam, batch, device, bce_w, dice_w):
    imgs = [s["img"] for s in batch]
    transformed, paddings = cellsam.prep_2(imgs, percentile=True)
    embeddings = cellsam.forward(transformed.to(device, non_blocking=True),
                                 return_preprocessed=False)

    total, n = torch.zeros((), device=device), 0
    for i, s in enumerate(batch):
        if len(s["boxes"]) == 0:
            continue
        logits = masks_for_boxes(cellsam, s["boxes"], s["img"].shape[-2:],
                                 paddings[i], embeddings[i])
        for lid, logit in zip(s["label_ids"].tolist(), logits):
            gt = torch.from_numpy((s["label_img"] == int(lid)).astype(np.float32)).to(device)
            total = total + (bce_w * F.binary_cross_entropy_with_logits(logit, gt)
                             + dice_w * dice_loss(logit, gt))
            n += 1
    return total / max(n, 1)


def run_epoch(cellsam, loader, device, bce_w, dice_w, optimizer=None):
    train = optimizer is not None
    cellsam.train() if train else cellsam.eval()
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    total = 0.0

    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        with torch.set_grad_enabled(train), \
             torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            loss = batch_loss(cellsam, batch, device, bce_w, dice_w)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in cellsam.parameters() if p.requires_grad], 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.item())
    return total / max(len(loader), 1)


def load_cellfinder_head(cellsam, ckpt):
    """The detector head from stage 3, so inference uses the same boxes."""
    sd = torch.load(os.path.expanduser(ckpt), map_location="cpu")
    sd = sd.get("state_dict", sd)
    for prefix in ("cellfinder.decode_head.", "module.cellfinder.decode_head."):
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            break
    cellsam.cellfinder.decode_head.load_state_dict(sd, strict=False)


def set_trainable(cellsam, train_neck=True):
    for p in cellsam.parameters():
        p.requires_grad = False
    for p in cellsam.model_cp.mask_decoder.parameters():
        p.requires_grad = True
    if train_neck and hasattr(cellsam.model_cp.image_encoder, "neck"):
        for p in cellsam.model_cp.image_encoder.neck.parameters():
            p.requires_grad = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_dir", required=True, help="folder with train/val_bboxes.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--init_cellsam_ckpt", required=True)
    ap.add_argument("--cellfinder_ckpt", required=True, help="cellfinder.pt from stage 3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr_decoder", type=float, default=1e-5)
    ap.add_argument("--lr_neck", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_instances_per_image", type=int, default=200)
    ap.add_argument("--sampler_weight", type=float, default=0.3)
    ap.add_argument("--bce_w", type=float, default=1.0)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_ckpt = os.path.join(a.outdir, "decoder.pt")
    loss_tsv = os.path.join(a.outdir, "decoder_losses.tsv")

    train_set = MaskDataset(os.path.join(a.split_dir, "train_bboxes.tsv"),
                            a.max_instances_per_image, a.seed)
    val_set = MaskDataset(os.path.join(a.split_dir, "val_bboxes.tsv"),
                          a.max_instances_per_image, a.seed + 1)
    print(f"  {len(train_set)} training images, {len(val_set)} validation images, "
          f"device {device}")

    # Same weighting as stage 3: augmented copies are drawn less often
    weights = torch.as_tensor([a.sampler_weight if "_aug" in uid else 1.0
                               for uid, _ in train_set.groups], dtype=torch.double)
    train_loader = DataLoader(train_set, batch_size=a.batch_size,
                              sampler=WeightedRandomSampler(weights, len(weights), True),
                              num_workers=a.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.num_workers, collate_fn=collate, pin_memory=True)

    cellsam = CellSAM({}).to(device)
    cellsam.load_state_dict(torch.load(os.path.expanduser(a.init_cellsam_ckpt),
                                       map_location="cpu"), strict=False)
    load_cellfinder_head(cellsam, a.cellfinder_ckpt)
    cellsam.adv_mode = True        # what CellSAM.from_pretrained sets, needed by predict
    set_trainable(cellsam)

    groups = [{"params": [p for p in cellsam.model_cp.mask_decoder.parameters()
                          if p.requires_grad], "lr": a.lr_decoder}]
    if hasattr(cellsam.model_cp.image_encoder, "neck"):
        groups.append({"params": [p for p in cellsam.model_cp.image_encoder.neck.parameters()
                                  if p.requires_grad], "lr": a.lr_neck})
    optimizer = AdamW(groups, weight_decay=a.weight_decay)

    best_val, rows = float("inf"), []
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(cellsam, train_loader, device, a.bce_w, a.dice_w, optimizer)
        val_loss = run_epoch(cellsam, val_loader, device, a.bce_w, a.dice_w)
        print(f"  epoch {epoch}/{a.epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
              f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(cellsam.state_dict(), best_ckpt)
            print(f"    new best, saved to {best_ckpt}", flush=True)

        rows.append({"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_loss, "best_val": best_val})
        pd.DataFrame(rows).to_csv(loss_tsv, sep="\t", index=False)

        plt.figure(figsize=(7, 4))
        plt.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows],
                 label="train", color="#5A949C")
        plt.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows],
                 label="validation", color="#D76C2D")
        plt.xlabel("epoch")
        plt.ylabel("BCE + Dice")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(a.outdir, "decoder_loss_curve.png"), dpi=200)
        plt.close()

    best_epoch = min(rows, key=lambda r: r["val_loss"])["epoch"]
    print(f"\n  done. best validation loss at epoch {best_epoch}: {best_val:.4f}")
    print(f"  checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
