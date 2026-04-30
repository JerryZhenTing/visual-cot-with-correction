"""
Reward functions for RL fine-tuning of the visual guidance policy.

reward = alpha_answer * answer_reward
       + alpha_iou    * iou_reward
       + alpha_format * format_reward
       - alpha_area   * area_penalty

answer_reward : 1.0 if VLM answer is correct, else 0.0
iou_reward    : IoU(pred_box, target_box)
format_reward : 1.0 if box is valid (non-degenerate), else 0.0
area_penalty  : area(pred_box), discourages predicting full image

Three preset reward modes:
  "combined"      : all four terms (default)
  "grounding"     : iou + format - area  (no VLM call needed — fast RL)
  "answer_only"   : answer_reward only
"""

from __future__ import annotations

from typing import Optional

import torch

# Default weights
ALPHA_ANSWER = 1.0
ALPHA_IOU    = 0.5
ALPHA_FORMAT = 0.2
ALPHA_AREA   = 0.05


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def compute_answer_reward(predicted_answer: Optional[str], ground_truth: str) -> float:
    """1.0 if predicted answer matches ground truth, else 0.0."""
    if predicted_answer is None:
        return 0.0
    return 1.0 if predicted_answer.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_iou_reward(pred_box: Optional[list], target_box: Optional[list]) -> float:
    """IoU between predicted and target box. 0.0 if either is None."""
    if pred_box is None or target_box is None:
        return 0.0
    px1, py1, px2, py2 = pred_box
    tx1, ty1, tx2, ty2 = target_box

    ix1 = max(px1, tx1); iy1 = max(py1, ty1)
    ix2 = min(px2, tx2); iy2 = min(py2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    if inter == 0.0:
        return 0.0

    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_t = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union  = area_p + area_t - inter
    return inter / union if union > 0.0 else 0.0


def compute_format_reward(pred_box: Optional[list]) -> float:
    """1.0 if box is a valid non-degenerate [xmin,ymin,xmax,ymax] in [0,1]."""
    if pred_box is None or len(pred_box) != 4:
        return 0.0
    xmin, ymin, xmax, ymax = pred_box
    if not all(0.0 <= v <= 1.0 for v in pred_box):
        return 0.0
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    return 1.0


def compute_area_penalty(pred_box: Optional[list]) -> float:
    """Area of the predicted box (fraction of image). 0.0 if None."""
    if pred_box is None:
        return 0.0
    return max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_reward(
    pred_box: Optional[list],
    target_box: Optional[list],
    ground_truth: str,
    predicted_answer: Optional[str] = None,
    reward_type: str = "combined",
    alpha_answer: float = ALPHA_ANSWER,
    alpha_iou: float    = ALPHA_IOU,
    alpha_format: float = ALPHA_FORMAT,
    alpha_area: float   = ALPHA_AREA,
) -> dict[str, float]:
    """
    Compute reward for a single (pred_box, vlm_answer) pair.

    Args:
        pred_box         : predicted box [xmin,ymin,xmax,ymax] or None
        target_box       : ground-truth relation box or None
        ground_truth     : "true" or "false"
        predicted_answer : VLM answer string or None (required for "combined"/"answer_only")
        reward_type      : "combined" | "grounding" | "answer_only"
        alpha_*          : reward weights
    Returns:
        dict with total reward and each component
    """
    r_answer = compute_answer_reward(predicted_answer, ground_truth)
    r_iou    = compute_iou_reward(pred_box, target_box)
    r_format = compute_format_reward(pred_box)
    r_area   = compute_area_penalty(pred_box)

    if reward_type == "grounding":
        total = r_iou + alpha_format * r_format - alpha_area * r_area
    elif reward_type == "answer_only":
        total = r_answer
    else:  # "combined"
        total = (alpha_answer * r_answer
                 + alpha_iou * r_iou
                 + alpha_format * r_format
                 - alpha_area * r_area)

    return {
        "total":          total,
        "answer_reward":  r_answer,
        "iou_reward":     r_iou,
        "format_reward":  r_format,
        "area_penalty":   r_area,
    }


def compute_rewards_batch(
    pred_boxes: list[Optional[list]],
    target_boxes: list[Optional[list]],
    ground_truths: list[str],
    predicted_answers: Optional[list[Optional[str]]] = None,
    reward_type: str = "combined",
    **kwargs,
) -> list[dict[str, float]]:
    """Compute rewards for a batch of examples."""
    if predicted_answers is None:
        predicted_answers = [None] * len(pred_boxes)
    return [
        compute_reward(pb, tb, gt, pa, reward_type=reward_type, **kwargs)
        for pb, tb, gt, pa in zip(pred_boxes, target_boxes, ground_truths, predicted_answers)
    ]


def rewards_to_tensor(rewards: list[dict], key: str = "total") -> torch.Tensor:
    """Extract one reward component as a float tensor."""
    return torch.tensor([r[key] for r in rewards], dtype=torch.float32)
