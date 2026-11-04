#!/usr/bin/env python3
"""
Stage 3: fine-tune CellFinder on gas vesicle clusters.

    python 01_train_cellfinder.py --split_dir 03_splits --outdir 04_runs/ft1

CellSAM finds objects in two steps: CellFinder draws a box around each object,
then SAM turns the box into a mask. CellFinder is an Anchor-DETR detector on
top of SAM's image encoder, and it was trained on whole cells. Gas vesicle
clusters are much smaller, so this is the part that has to be retrained.

Only CellFinder's decode head is trained. The SAM encoder and the mask decoder
stay frozen, which is why the run fits on one GPU.

The checkpoint with the lowest validation loss is kept rather than the last
one. With around a hundred images the model starts memorising them after a few
epochs, and the loss curve written alongside shows when that happens.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from lib_io import boxes_to_cxcywh_norm, read_image_any, to_uint8_rgb

import cellSAM
from cellSAM.sam_inference import CellSAM

sys.path.insert(0, os.path.dirname(cellSAM.__file__))
import cellSAM.AnchorDETR.models.anchor_detr as anchor_detr
from cellSAM.AnchorDETR.models.anchor_detr import SetCriterion
from cellSAM.AnchorDETR.models.matcher import build_matcher


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    """Replaces the version shipped with Anchor-DETR.

    The original reduces over the wrong axis for our batch shapes, which makes
    the classification term scale with the number of queries instead of the
    number of boxes.
    """
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean(1).sum() / num_boxes


anchor_detr.sigmoid_focal_loss = sigmoid_focal_loss


class BoxDataset(Dataset):
    """One item per image. Rows without a box are negative examples and yield
    zero targets, which is how DETR learns "no object here"."""

    def __init__(self, tsv):
        self.groups = list(pd.read_csv(tsv, sep="\t", engine="python").groupby("uid"))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, g = self.groups[idx]
        img = read_image_any(g["image_path"].iloc[0])
        g = g.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
        h, w = img.shape
        boxes = boxes_to_cxcywh_norm(g, width=w, height=h)
        target = {"labels": torch.zeros(len(boxes), dtype=torch.int64),
                  "boxes": torch.from_numpy(boxes)}
        return to_uint8_rgb(img), target


def collate(batch):
    # No torch.stack: 90 and 270 degree rotations are 960x720 instead of
    # 720x960. CellSAM's preprocessing takes a list and resizes each image.
    xs, targets = zip(*batch)
    return list(xs), list(targets)


def sampler_weights(dataset, aug_weight, marker="_aug"):
    return torch.as_tensor([aug_weight if marker in str(uid) else 1.0
                            for uid, _ in dataset.groups], dtype=torch.double)


def run_epoch(model, cellsam, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    total = 0.0

    for xs, targets in tqdm(loader, leave=False, desc="train" if train else "val"):
        xs = cellsam.sam_bbox_preprocessing(list(xs), percentile=False)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with torch.set_grad_enabled(train), \
             torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            loss_dict = criterion(model(xs), targets)
            loss = sum(loss_dict[k] * criterion.weight_dict[k]
                       for k in loss_dict if k in criterion.weight_dict)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total += loss.item()
    return total / max(len(loader), 1)


def plot_losses(tsv, png):
    df = pd.read_csv(tsv, sep="\t")
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train", color="#5A949C")
    plt.plot(df["epoch"], df["val_loss"], label="validation", color="#D76C2D")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png, dpi=200)
    plt.close()


def set_backbone_trainable(head, trainable):
    backbone = getattr(head, "backbone", None)
    if backbone is None:
        print("  no backbone attribute on the decode head, nothing to freeze")
        return
    for p in backbone.parameters():
        p.requires_grad = trainable


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_dir", required=True, help="folder with train/val_bboxes.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--init_cellsam_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--sampler_weight", type=float, default=0.3,
                    help="how often augmented images are drawn, against 1.0 "
                         "for an original")
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_tsv = os.path.join(a.outdir, "losses.tsv")
    best_ckpt = os.path.join(a.outdir, "cellfinder.pt")

    train_set = BoxDataset(os.path.join(a.split_dir, "train_bboxes.tsv"))
    val_set = BoxDataset(os.path.join(a.split_dir, "val_bboxes.tsv"))
    print(f"  {len(train_set)} training images, {len(val_set)} validation images, "
          f"device {device}")

    weights = sampler_weights(train_set, a.sampler_weight)
    train_loader = DataLoader(train_set, batch_size=a.batch_size,
                              sampler=WeightedRandomSampler(weights, len(weights), True),
                              num_workers=a.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.num_workers, collate_fn=collate, pin_memory=True)

    # Model. Everything except CellFinder's decode head stays frozen.
    cellsam = CellSAM({}).to(device).eval()
    cellsam.load_state_dict(torch.load(os.path.expanduser(a.init_cellsam_ckpt),
                                       map_location="cpu"), strict=False)
    for p in list(cellsam.model.parameters()) + list(cellsam.model_cp.parameters()):
        p.requires_grad = False

    head = cellsam.cellfinder.decode_head
    head.train()
    if a.freeze_backbone_epochs > 0:
        set_backbone_trainable(head, False)
        print(f"  backbone frozen for the first {a.freeze_backbone_epochs} epochs")

    # DETR set criterion: classification, L1 on the box, and generalised IoU
    class Args:
        set_cost_class, set_cost_bbox, set_cost_giou, num_classes = 1.0, 5.0, 2.0, 2

    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    # Anchor-DETR supervises every decoder layer, so the same weights are
    # repeated for the five intermediate ones.
    weight_dict.update({f"{k}_{i}": v for i in range(5) for k, v in list(weight_dict.items())})
    criterion = SetCriterion(Args.num_classes, build_matcher(Args()), weight_dict,
                             ["labels", "boxes"]).to(device)

    def param_groups():
        return [
            {"params": [p for n, p in head.named_parameters()
                        if "backbone" not in n and p.requires_grad], "lr": a.lr},
            {"params": [p for n, p in head.named_parameters()
                        if "backbone" in n and p.requires_grad], "lr": a.lr_backbone},
        ]

    optimizer = AdamW(param_groups(), lr=a.lr, weight_decay=a.weight_decay)
    best_val, rows = float("inf"), []

    for epoch in range(1, a.epochs + 1):
        if a.freeze_backbone_epochs > 0 and epoch == a.freeze_backbone_epochs + 1:
            set_backbone_trainable(head, True)
            optimizer = AdamW(param_groups(), lr=a.lr, weight_decay=a.weight_decay)
            print(f"  backbone unfrozen at epoch {epoch}")

        t0 = time.time()
        train_loss = run_epoch(head, cellsam, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(head, cellsam, val_loader, criterion, device)
        print(f"  epoch {epoch}/{a.epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
              f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), best_ckpt)
            print(f"    new best, saved to {best_ckpt}", flush=True)

        rows.append({"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_loss, "best_val": best_val})
        pd.DataFrame(rows).to_csv(loss_tsv, sep="\t", index=False)
        plot_losses(loss_tsv, os.path.join(a.outdir, "loss_curve.png"))

    best_epoch = min(rows, key=lambda r: r["val_loss"])["epoch"]
    print(f"\n  done. best validation loss at epoch {best_epoch}: {best_val:.4f}")
    print(f"  checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
