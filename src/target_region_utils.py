"""
Region computation utilities for targeted adversarial perturbations.

All boxes use normalized [xmin, ymin, xmax, ymax] in [0, 1], matching the
convention used throughout this project.

VSR object boxes (obj1_bbox / obj2_bbox) come from vsr_with_bboxes.jsonl
in COCO format [x, y, w, h] absolute pixels. This module converts them to
normalized xyxy and derives target regions from them.

If either object box is missing the example cannot be perturbed and should
be skipped — call get_boxes_for_example() which returns a skip_reason.
"""

from __future__ import annotations

import random
from typing import Optional

from load_vsr import VSRExample, ANNOTATION_BBOX_FORMAT


# ---------------------------------------------------------------------------
# COCO → xyxy conversion (mirrors load_vsr._bbox_to_xyxy)
# ---------------------------------------------------------------------------

def _coco_to_xyxy(b: list) -> list:
    if ANNOTATION_BBOX_FORMAT == "coco":
        x, y, w, h = b
        return [x, y, x + w, y + h]
    return list(b)


def _normalize_xyxy(xyxy: list, W: int, H: int) -> list:
    return [
        max(0.0, min(1.0, xyxy[0] / W)),
        max(0.0, min(1.0, xyxy[1] / H)),
        max(0.0, min(1.0, xyxy[2] / W)),
        max(0.0, min(1.0, xyxy[3] / H)),
    ]


# ---------------------------------------------------------------------------
# Per-object box helpers
# ---------------------------------------------------------------------------

def get_obj1_box_normalized(ex: VSRExample) -> Optional[list]:
    """obj1 as normalized [xmin, ymin, xmax, ymax], or None."""
    if ex.obj1_bbox is None:
        return None
    return _normalize_xyxy(_coco_to_xyxy(ex.obj1_bbox), ex.image_width, ex.image_height)


def get_obj2_box_normalized(ex: VSRExample) -> Optional[list]:
    """obj2 as normalized [xmin, ymin, xmax, ymax], or None."""
    if ex.obj2_bbox is None:
        return None
    return _normalize_xyxy(_coco_to_xyxy(ex.obj2_bbox), ex.image_width, ex.image_height)


# ---------------------------------------------------------------------------
# Derived regions
# ---------------------------------------------------------------------------

def get_union_box(box1: list, box2: list) -> list:
    """Axis-aligned union of two normalized boxes."""
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    ]


def get_padded_box(box: list, pad: float = 0.05) -> list:
    """Expand a normalized box by `pad` on each side, clipped to [0, 1]."""
    return [
        max(0.0, box[0] - pad),
        max(0.0, box[1] - pad),
        min(1.0, box[2] + pad),
        min(1.0, box[3] + pad),
    ]


def get_relation_region(box1: list, box2: list, pad: float = 0.05) -> list:
    """
    Relation-critical region: padded union of the two object boxes.

    Approximates the region containing both objects and the space between them.
    The padding ensures we capture slightly more context than the tight union.
    """
    return get_padded_box(get_union_box(box1, box2), pad=pad)


# ---------------------------------------------------------------------------
# Area and overlap helpers
# ---------------------------------------------------------------------------

def box_area(box: list) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def boxes_overlap(a: list, b: list) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# ---------------------------------------------------------------------------
# Distractor placement helpers
# ---------------------------------------------------------------------------

def find_distractor_source_patch(
    avoid_boxes: list[list],
    patch_w: float,
    patch_h: float,
    seed: Optional[int] = None,
    max_tries: int = 200,
) -> Optional[list]:
    """
    Find a normalized source patch region that does not overlap any avoid_boxes.

    Samples candidate top-left corners uniformly at random.

    Args:
        avoid_boxes: list of normalized xyxy boxes to avoid
        patch_w:     patch width as fraction of image width
        patch_h:     patch height as fraction of image height
        seed:        RNG seed
        max_tries:   maximum sampling attempts before returning None

    Returns:
        [xmin, ymin, xmax, ymax] or None if no non-overlapping region found
    """
    rng = random.Random(seed)
    for _ in range(max_tries):
        x0 = rng.uniform(0.0, max(0.0, 1.0 - patch_w))
        y0 = rng.uniform(0.0, max(0.0, 1.0 - patch_h))
        candidate = [x0, y0, x0 + patch_w, y0 + patch_h]
        if not any(boxes_overlap(candidate, avoid) for avoid in avoid_boxes):
            return candidate
    return None


def distractor_destination_near_box(
    near_box: list,
    patch_w: float,
    patch_h: float,
    seed: Optional[int] = None,
    offset_range: float = 0.12,
) -> list:
    """
    Place a distractor patch near `near_box` with seed-controlled jitter.

    The patch center is placed at the center of near_box plus a random offset.
    Clipped to keep the patch fully inside [0, 1].

    Args:
        near_box:     normalized xyxy box to place near
        patch_w/h:    patch size as fraction of image dimensions
        seed:         RNG seed
        offset_range: max random offset as fraction of image dimensions

    Returns:
        [xmin, ymin, xmax, ymax] clipped to [0, 1]
    """
    rng = random.Random(seed)
    cx = (near_box[0] + near_box[2]) / 2.0
    cy = (near_box[1] + near_box[3]) / 2.0
    ox = rng.uniform(-offset_range, offset_range)
    oy = rng.uniform(-offset_range, offset_range)
    x0 = max(0.0, min(1.0 - patch_w, cx + ox - patch_w / 2.0))
    y0 = max(0.0, min(1.0 - patch_h, cy + oy - patch_h / 2.0))
    return [x0, y0, min(1.0, x0 + patch_w), min(1.0, y0 + patch_h)]


# ---------------------------------------------------------------------------
# Main entry point for generation
# ---------------------------------------------------------------------------

def get_boxes_for_example(
    ex: VSRExample,
) -> tuple[Optional[list], Optional[list], Optional[str]]:
    """
    Return (obj1_box_norm, obj2_box_norm, skip_reason).

    skip_reason is None when both boxes are valid. When skip_reason is set,
    obj1_box_norm and obj2_box_norm are both None.
    """
    b1 = get_obj1_box_normalized(ex)
    b2 = get_obj2_box_normalized(ex)

    if b1 is None and b2 is None:
        return None, None, "both obj1_bbox and obj2_bbox missing"
    if b1 is None:
        return None, None, "obj1_bbox missing"
    if b2 is None:
        return None, None, "obj2_bbox missing"
    if box_area(b1) <= 0:
        return None, None, f"obj1_bbox has zero area after normalization: {b1}"
    if box_area(b2) <= 0:
        return None, None, f"obj2_bbox has zero area after normalization: {b2}"

    return b1, b2, None
