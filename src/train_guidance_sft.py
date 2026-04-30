"""
Supervised training of the visual guidance policy.

Trains the BoxHead of GuidancePolicy to predict the relation-relevant
bounding box for each VSR example, using a combined L1 + GIoU + area loss.
The CLIP encoders remain frozen throughout.

CLI:
    python src/train_guidance_sft.py \\
        --data data/vsr_guidance.json \\
        --epochs 10 \\
        --batch-size 32 \\
        --lr 1e-4 \\
        --seed 42 \\
        --output-dir ../checkpoints/guidance_sft

Outputs:
    ../checkpoints/guidance_sft/best.pt      (best val mean IoU)
    ../checkpoints/guidance_sft/final.pt
    ../checkpoints/guidance_sft/train_log.json
    ../checkpoints/guidance_sft/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guidance_dataset import VSRGuidanceDataset, _make_splits
from guidance_losses import guidance_loss
from guidance_model import GuidancePolicy, collate_fn
from metrics import rsa_at_threshold

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ---------------------------------------------------------------------------
# Collate helper that binds the processor
# ---------------------------------------------------------------------------

def make_collate(processor, device):
    def _collate(batch):
        # Filter out items with missing images
        batch = [b for b in batch if b["image"] is not None]
        if not batch:
            return None
        return collate_fn(batch, processor, device)
    return _collate


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ious, rsa5, rsa25, areas, n_invalid = [], [], [], [], 0
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue

        pred = model(
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
        )
        target = batch["target_boxes"]

        loss_dict = guidance_loss(pred, target)
        total_loss += loss_dict["total"].item()
        n_batches += 1

        pred_list   = pred.cpu().tolist()
        target_list = target.cpu().tolist()

        for pb, tb in zip(pred_list, target_list):
            # IoU
            iou_val = _box_iou_scalar(pb, tb)
            ious.append(iou_val)
            rsa5.append(1.0 if iou_val >= 0.5 else 0.0)
            rsa25.append(1.0 if iou_val >= 0.25 else 0.0)

            # Area
            area = (pb[2] - pb[0]) * (pb[3] - pb[1])
            areas.append(max(0.0, area))

            # Validity: xmax > xmin and ymax > ymin
            if pb[2] <= pb[0] or pb[3] <= pb[1]:
                n_invalid += 1

    n = len(ious) or 1
    return {
        "val_loss":       total_loss / max(n_batches, 1),
        "mean_iou":       sum(ious) / n,
        "rsa_50":         sum(rsa5) / n,
        "rsa_25":         sum(rsa25) / n,
        "mean_area":      sum(areas) / n,
        "invalid_rate":   n_invalid / n,
        "large_box_rate": sum(1 for a in areas if a > 0.5) / n,
    }


def _box_iou_scalar(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-7)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Model
    print(f"Loading CLIP model: {args.clip_model}")
    model = GuidancePolicy.from_pretrained(args.clip_model).to(device)
    processor = model.processor

    # Dataset
    cache_abs = args.data if os.path.isabs(args.data) else os.path.join(_ROOT, args.data)
    all_ds = VSRGuidanceDataset.from_cache(cache_abs, split="all", seed=args.seed)
    splits = _make_splits(all_ds.examples, args.seed)
    train_ds = VSRGuidanceDataset(splits["train"])
    val_ds   = VSRGuidanceDataset(splits["val"])
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    collate = make_collate(processor, device)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    # Optimizer and LR schedule (only BoxHead parameters)
    optimizer = torch.optim.AdamW(
        model.box_head.parameters(), lr=args.lr, weight_decay=1e-4
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_iou = -1.0
    train_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            if batch is None:
                continue

            optimizer.zero_grad()
            pred   = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
            target = batch["target_boxes"]

            loss_dict = guidance_loss(
                pred, target,
                lambda_l1=args.lambda_l1,
                lambda_iou=args.lambda_iou,
                lambda_area=args.lambda_area,
            )
            loss = loss_dict["total"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.box_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0

        log_entry = {
            "epoch":    epoch,
            "train_loss": avg_loss,
            **val_metrics,
            "lr":       scheduler.get_last_lr()[0],
            "time_s":   elapsed,
        }
        train_log.append(log_entry)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"val_iou={val_metrics['mean_iou']:.4f}  "
            f"rsa50={val_metrics['rsa_50']:.4f}  "
            f"rsa25={val_metrics['rsa_25']:.4f}  "
            f"area={val_metrics['mean_area']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            model.save_checkpoint(
                os.path.join(args.output_dir, "best.pt"),
                meta={"epoch": epoch, **val_metrics},
            )
            print(f"  → New best (IoU={best_iou:.4f}), saved best.pt")

        with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
            json.dump(train_log, f, indent=2)

    model.save_checkpoint(
        os.path.join(args.output_dir, "final.pt"),
        meta={"epoch": args.epochs, **validate(model, val_loader, device)},
    )
    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoints in: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Supervised training of visual guidance policy.")
    p.add_argument("--data",         default="data/vsr_guidance_full.json",
                   help="Path to cached guidance dataset JSON")
    p.add_argument("--clip-model",   default="openai/clip-vit-base-patch32")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--lambda-l1",    type=float, default=1.0)
    p.add_argument("--lambda-iou",   type=float, default=1.0)
    p.add_argument("--lambda-area",  type=float, default=0.05)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--output-dir",   default="../checkpoints/guidance_sft")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
