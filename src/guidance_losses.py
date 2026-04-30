"""
Supervised losses for training the visual guidance policy.

Loss = lambda_l1 * L1(pred, target)
     + lambda_iou * (1 - GIoU(pred, target))
     + lambda_area * area_penalty(pred)

GIoU (Generalized IoU) is preferred over plain IoU loss because it provides
a gradient even when boxes do not overlap, which helps early in training.

All boxes are expected in normalized [xmin, ymin, xmax, ymax] format.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_LAMBDA_L1   = 1.0
DEFAULT_LAMBDA_IOU  = 1.0
DEFAULT_LAMBDA_AREA = 0.05


# ---------------------------------------------------------------------------
# GIoU
# ---------------------------------------------------------------------------

def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU loss for batched boxes.

    Args:
        pred   : (B, 4) predicted boxes [xmin,ymin,xmax,ymax]
        target : (B, 4) target boxes    [xmin,ymin,xmax,ymax]
    Returns:
        (B,) per-example GIoU loss values (mean over batch when reduced)
    """
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # Intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

    # Union
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + 1e-7

    iou = inter / union

    # Smallest enclosing box
    cx1 = torch.min(px1, tx1)
    cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2)
    cy2 = torch.max(py2, ty2)
    enclosing = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-7

    giou = iou - (enclosing - union) / enclosing
    return 1.0 - giou   # loss in [0, 2]


# ---------------------------------------------------------------------------
# Area penalty
# ---------------------------------------------------------------------------

def area_penalty(pred: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Penalize predicted boxes that are too large.

    penalty = max(0, area(pred) - margin)

    With margin=0 this simply penalizes any area, discouraging the policy
    from predicting the entire image.

    Args:
        pred   : (B, 4) predicted boxes
        margin : allowed area before penalty kicks in (default 0 = penalize all area)
    Returns:
        (B,) per-example penalty values
    """
    w = (pred[:, 2] - pred[:, 0]).clamp(min=0)
    h = (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area = w * h
    return torch.clamp(area - margin, min=0.0)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def guidance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_l1: float   = DEFAULT_LAMBDA_L1,
    lambda_iou: float  = DEFAULT_LAMBDA_IOU,
    lambda_area: float = DEFAULT_LAMBDA_AREA,
    area_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Compute the combined guidance loss.

    Args:
        pred        : (B, 4) predicted boxes
        target      : (B, 4) target boxes
        lambda_l1   : weight for L1 coordinate loss
        lambda_iou  : weight for GIoU loss
        lambda_area : weight for area penalty
        area_margin : area fraction above which penalty kicks in
    Returns:
        dict with keys: total, l1, giou, area (all scalar tensors)
    """
    l1   = F.l1_loss(pred, target, reduction="mean")
    giou = giou_loss(pred, target).mean()
    area = area_penalty(pred, margin=area_margin).mean()

    total = lambda_l1 * l1 + lambda_iou * giou + lambda_area * area

    return {
        "total": total,
        "l1":    l1,
        "giou":  giou,
        "area":  area,
    }
