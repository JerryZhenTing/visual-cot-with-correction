"""
Evaluation metrics for the Visual CoT robustness project.

Functions:
  final_answer_accuracy  – fraction of correctly predicted answers
  valid_box_rate         – fraction of structurally valid predicted boxes
  iou                    – intersection over union for two boxes
  compute_rsa            – region-sensitive accuracy (IoU >= threshold)
  mean_rsa               – mean RSA across a results list
  denormalize_box        – convert normalized [0,1] box to pixel coordinates
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Answer accuracy
# ---------------------------------------------------------------------------

def final_answer_accuracy(results: list[dict]) -> float:
    """
    Fraction of examples where parsed_answer == ground_truth.

    Uses the "answer_correct" bool field set by the run scripts.

    Args:
        results: list of result dicts
    Returns:
        float in [0, 1]
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("answer_correct", False))
    return correct / len(results)


# ---------------------------------------------------------------------------
# Box quality
# ---------------------------------------------------------------------------

def valid_box_rate(results: list[dict]) -> float:
    """
    Fraction of examples that have a structurally valid predicted box.

    Uses the "box_valid" bool field set by parse_outputs.parse_box().

    Args:
        results: list of result dicts
    Returns:
        float in [0, 1]
    """
    if not results:
        return 0.0
    valid = sum(1 for r in results if r.get("box_valid", False))
    return valid / len(results)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection over Union between two boxes.

    Boxes must be in [xmin, ymin, xmax, ymax] format and share the same
    coordinate system (both normalized or both absolute pixels).

    Args:
        box_a: [xmin, ymin, xmax, ymax]
        box_b: [xmin, ymin, xmax, ymax]
    Returns:
        float in [0, 1]
    """
    # Clip to [0, 1] — COCO annotations near image edges can exceed bounds
    def _clip(b):
        x1, y1, x2, y2 = b
        return (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2)))
    ax1, ay1, ax2, ay2 = _clip(box_a)
    bx1, by1, bx2, by2 = _clip(box_b)

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    if intersection == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# RSA
# ---------------------------------------------------------------------------

def compute_rsa(
    pred_box: Optional[list],
    target_box: Optional[list],
    threshold: float = 0.5,
) -> Optional[float]:
    """
    Region Sensitive Accuracy for a single example.

    Definition:
        RSA = 1.0  if IoU(pred_box, target_box) >= threshold
        RSA = 0.0  otherwise
        RSA = None if either box is unavailable

    target_box is the union of the two relevant object bounding boxes
    (computed by VSRExample.target_box_normalized).  Both boxes must be
    in the same normalized [0,1] coordinate system.

    Args:
        pred_box   : predicted box [xmin,ymin,xmax,ymax] in [0,1], or None
        target_box : target box  [xmin,ymin,xmax,ymax] in [0,1], or None
        threshold  : IoU threshold (default 0.5)
    Returns:
        1.0, 0.0, or None
    """
    if pred_box is None or target_box is None:
        return None
    score = iou(pred_box, target_box)
    return 1.0 if score >= threshold else 0.0


def box_revision_rate(results: list[dict]) -> float:
    """
    Fraction of examples where pass2 box differs from pass1 box.
    Uses "box_revised" bool field set by run_visual_cot_verification.py.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("box_revised", False)) / len(results)


def answer_flip_rate(results: list[dict]) -> float:
    """
    Fraction of examples where pass2 answer differs from pass1 answer.
    Uses "answer_flipped" bool field.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("answer_flipped", False)) / len(results)


def recovery_rate(results: list[dict]) -> Optional[float]:
    """
    Among examples where pass1 answer is WRONG, fraction where pass2 becomes correct.
    Returns None if pass1 was never wrong.
    Uses "pass1_answer_correct" and "pass2_answer_correct" fields.
    """
    wrong_in_pass1 = [r for r in results if not r.get("pass1_answer_correct", True)]
    if not wrong_in_pass1:
        return None
    recovered = sum(1 for r in wrong_in_pass1 if r.get("pass2_answer_correct", False))
    return recovered / len(wrong_in_pass1)


def answer_change_rate(results: list[dict]) -> float:
    """
    Fraction of examples where verification changed the answer vs. initial Visual CoT.

    Uses the "answer_changed" bool field set by run_verification.py.
    """
    if not results:
        return 0.0
    changed = sum(1 for r in results if r.get("answer_changed", False))
    return changed / len(results)


# ---------------------------------------------------------------------------
# Performance Drop Rate (PDR)
# ---------------------------------------------------------------------------

def performance_drop_rate(
    clean_faa: Optional[float],
    corrupted_faas: list[float],
) -> Optional[float]:
    """
    Performance Drop Rate: relative FAA drop from clean to a corruption family.

    Definition:
        PDR = (clean_FAA - mean_corrupted_FAA) / clean_FAA

    A positive PDR means performance degraded under corruption.
    PDR = 0 means no drop.  PDR < 0 means performance improved (unusual).

    Args:
        clean_faa:      FAA on clean images
        corrupted_faas: list of FAA values for one corruption family
                        (e.g. the three blur severities)
    Returns:
        PDR as a float, or None if clean_faa is None or zero, or no corrupted values
    """
    if clean_faa is None or clean_faa == 0.0 or not corrupted_faas:
        return None
    mean_corrupted = sum(corrupted_faas) / len(corrupted_faas)
    return (clean_faa - mean_corrupted) / clean_faa


def mean_rsa(results: list[dict]) -> Optional[float]:
    """
    Mean RSA across all results that have a non-None rsa value.

    Returns None if no RSA values are available (e.g., no target boxes).
    """
    values = [r["rsa"] for r in results if r.get("rsa") is not None]
    if not values:
        return None
    return sum(values) / len(values)


def rsa_at_threshold(
    pred_box: Optional[list],
    target_box: Optional[list],
    threshold: float,
) -> Optional[float]:
    """RSA for a single example at an arbitrary IoU threshold."""
    if pred_box is None or target_box is None:
        return None
    return 1.0 if iou(pred_box, target_box) >= threshold else 0.0


def mean_iou(results: list[dict], iou_key: str = "iou") -> Optional[float]:
    """Mean IoU across results that have a non-None value for iou_key."""
    values = [r[iou_key] for r in results if r.get(iou_key) is not None]
    return sum(values) / len(values) if values else None


def box_area(box: list) -> float:
    """Area of a normalized [xmin,ymin,xmax,ymax] box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def invalid_box_rate(results: list[dict], box_key: str = "predicted_box") -> float:
    """Fraction of results where the predicted box is None."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get(box_key) is None) / len(results)


def coverage_too_large_rate(
    results: list[dict],
    box_key: str = "predicted_box",
    threshold: float = 0.5,
) -> float:
    """Fraction of results where predicted box covers more than `threshold` of the image."""
    if not results:
        return 0.0
    return sum(
        1 for r in results
        if r.get(box_key) is not None and box_area(r[box_key]) > threshold
    ) / len(results)


# ---------------------------------------------------------------------------
# Coordinate utility
# ---------------------------------------------------------------------------

def denormalize_box(
    box_normalized: list,
    image_width: int,
    image_height: int,
) -> list[float]:
    """
    Convert a normalized [0,1] box to absolute pixel coordinates.

    Args:
        box_normalized : [xmin, ymin, xmax, ymax] in [0, 1]
        image_width    : image width in pixels
        image_height   : image height in pixels
    Returns:
        [xmin, ymin, xmax, ymax] in pixels
    """
    xmin, ymin, xmax, ymax = box_normalized
    return [
        xmin * image_width,
        ymin * image_height,
        xmax * image_width,
        ymax * image_height,
    ]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # IoU: perfect overlap
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0

    # IoU: no overlap
    assert iou([0, 0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]) == 0.0

    # IoU: partial overlap
    score = iou([0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75])
    assert abs(score - 1/7) < 1e-6, f"Got {score}"

    # RSA
    assert compute_rsa([0, 0, 1, 1], [0, 0, 1, 1], threshold=0.5) == 1.0
    assert compute_rsa([0, 0, 0.1, 0.1], [0.9, 0.9, 1.0, 1.0], threshold=0.5) == 0.0
    assert compute_rsa(None, [0, 0, 1, 1]) is None

    # mean_rsa with some Nones
    results = [{"rsa": 1.0}, {"rsa": 0.0}, {"rsa": None}, {"rsa": 1.0}]
    assert abs(mean_rsa(results) - 2 / 3) < 1e-9

    # FAA
    r = [{"answer_correct": True}, {"answer_correct": False}, {"answer_correct": True}]
    assert abs(final_answer_accuracy(r) - 2 / 3) < 1e-9

    print("All metrics smoke tests passed.")
