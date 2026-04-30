"""
Rule-based targeted perturbations for VSR evidence-disruption evaluation.

Three perturbation families:
  1. Object-centered occlusion  — covers part of one specific object's box
  2. Relation-critical masking  — covers part of the union/relation region
  3. Distractor patch insertion — pastes a background patch near a relevant object

Design principles:
  - All perturbations preserve image geometry (W, H unchanged)
  - Reference bounding boxes remain in the same coordinate system
  - RSA can still be computed for all targeted perturbations (rsa_valid=True)
  - Perturbations are deterministic given a seed
  - Operations use normalized [xmin, ymin, xmax, ymax] in [0, 1]

Unlike common corruptions (blur, noise, rotation) which degrade the image globally,
these perturbations selectively disrupt the visual evidence needed to answer the
specific VSR question for each example.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

# Fraction of the target region's area that is covered by occlusion or masking.
SEVERITY_TO_RATIO = {
    "low":    0.25,
    "medium": 0.50,
    "high":   0.75,
}

# For distractor insertion: patch area as a fraction of the relation-region area.
DISTRACTOR_SEVERITY_TO_RATIO = {
    "low":    0.15,
    "medium": 0.25,
    "high":   0.35,
}

VALID_SEVERITIES = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Perturbation spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerturbationSpec:
    """Describes one targeted perturbation (type × target × severity)."""
    perturbation_type: str  # "occlusion" | "mask" | "distractor"
    target: str             # "object_1" | "object_2" | "union" | "near_obj1" | "near_obj2"
    severity: str           # "low" | "medium" | "high"
    display_name: str       # compact label, e.g. "occlusion-obj1-high"
    # Geometry is preserved for all targeted perturbations → RSA always valid.
    rsa_valid: bool = True
    subdir: str = ""        # relative path under data/targeted/


def get_all_targeted_perturbation_specs() -> list[PerturbationSpec]:
    """
    Return the full list of targeted perturbation specs (15 total):
      Occlusion  × {object_1, object_2} × {low, medium, high} = 6
      Mask       × {union}              × {low, medium, high} = 3
      Distractor × {near_obj1, near_obj2} × {low, medium, high} = 6
    """
    specs = []
    for severity in VALID_SEVERITIES:
        specs.append(PerturbationSpec(
            perturbation_type="occlusion", target="object_1", severity=severity,
            display_name=f"occlusion-obj1-{severity}",
            subdir=f"occlusion/object_1/{severity}",
        ))
        specs.append(PerturbationSpec(
            perturbation_type="occlusion", target="object_2", severity=severity,
            display_name=f"occlusion-obj2-{severity}",
            subdir=f"occlusion/object_2/{severity}",
        ))
        specs.append(PerturbationSpec(
            perturbation_type="mask", target="union", severity=severity,
            display_name=f"mask-union-{severity}",
            subdir=f"mask/union/{severity}",
        ))
        specs.append(PerturbationSpec(
            perturbation_type="distractor", target="near_obj1", severity=severity,
            display_name=f"distractor-nearobj1-{severity}",
            subdir=f"distractor/near_obj1/{severity}",
        ))
        specs.append(PerturbationSpec(
            perturbation_type="distractor", target="near_obj2", severity=severity,
            display_name=f"distractor-nearobj2-{severity}",
            subdir=f"distractor/near_obj2/{severity}",
        ))
    return specs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _denorm(box: list, W: int, H: int) -> Tuple[int, int, int, int]:
    """Normalized [0,1] xyxy → integer pixel coords."""
    return (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))


def _fill_rgb(fill_mode: str) -> Tuple[int, int, int]:
    return (128, 128, 128) if fill_mode == "gray" else (0, 0, 0)


# ---------------------------------------------------------------------------
# 1. Object-centered occlusion
# ---------------------------------------------------------------------------

def apply_object_occlusion(
    image: Image.Image,
    target_box: list,
    severity: str,
    fill_mode: str = "black",
    seed: Optional[int] = None,
) -> Tuple[Image.Image, dict]:
    """
    Cover a fraction of the target object box with a solid rectangle.

    The occluding rectangle covers `ratio` of the target box area (ratio = SEVERITY_TO_RATIO[severity]).
    Both dimensions are scaled by sqrt(ratio) to preserve the box's aspect ratio.
    Placement within the target box is seed-controlled random.

    Image geometry is preserved → RSA is valid.

    Returns:
        (modified_image, metadata_dict)
    """
    ratio = SEVERITY_TO_RATIO[severity]
    W, H = image.size
    tx1, ty1, tx2, ty2 = _denorm(target_box, W, H)
    bw, bh = tx2 - tx1, ty2 - ty1

    if bw <= 0 or bh <= 0:
        return image.copy(), {"occluded_region": None, "skip_reason": "zero-area target box"}

    # Scale both sides by sqrt(ratio) to cover `ratio` fraction of area
    ow = max(1, int(bw * ratio ** 0.5))
    oh = max(1, int(bh * ratio ** 0.5))

    rng = random.Random(seed)
    ox = rng.randint(0, max(0, bw - ow))
    oy = rng.randint(0, max(0, bh - oh))
    x1, y1 = tx1 + ox, ty1 + oy
    x2, y2 = x1 + ow, y1 + oh

    img = image.copy().convert("RGB")
    ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], fill=_fill_rgb(fill_mode))

    return img, {
        "target_box":       target_box,
        "occluded_region":  [x1 / W, y1 / H, x2 / W, y2 / H],
        "coverage_ratio":   ratio,
        "severity":         severity,
        "fill_mode":        fill_mode,
        "seed":             seed,
        "skip_reason":      None,
    }


# ---------------------------------------------------------------------------
# 2. Relation-critical masking
# ---------------------------------------------------------------------------

def apply_relation_mask(
    image: Image.Image,
    relation_box: list,
    severity: str,
    fill_mode: str = "black",
    seed: Optional[int] = None,
) -> Tuple[Image.Image, dict]:
    """
    Cover a fraction of the relation-critical region (padded union of object boxes).

    Works identically to occlusion but targets the relation region rather than
    a single object. Disrupts the spatial evidence between both objects.

    Image geometry is preserved → RSA is valid.

    Returns:
        (modified_image, metadata_dict)
    """
    ratio = SEVERITY_TO_RATIO[severity]
    W, H = image.size
    rx1, ry1, rx2, ry2 = _denorm(relation_box, W, H)
    rw, rh = rx2 - rx1, ry2 - ry1

    if rw <= 0 or rh <= 0:
        return image.copy(), {"masked_region": None, "skip_reason": "zero-area relation box"}

    mw = max(1, int(rw * ratio ** 0.5))
    mh = max(1, int(rh * ratio ** 0.5))

    rng = random.Random(seed)
    ox = rng.randint(0, max(0, rw - mw))
    oy = rng.randint(0, max(0, rh - mh))
    x1, y1 = rx1 + ox, ry1 + oy
    x2, y2 = x1 + mw, y1 + mh

    img = image.copy().convert("RGB")
    ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], fill=_fill_rgb(fill_mode))

    return img, {
        "relation_box":  relation_box,
        "masked_region": [x1 / W, y1 / H, x2 / W, y2 / H],
        "coverage_ratio": ratio,
        "severity":      severity,
        "fill_mode":     fill_mode,
        "seed":          seed,
        "skip_reason":   None,
    }


# ---------------------------------------------------------------------------
# 3. Distractor patch insertion
# ---------------------------------------------------------------------------

def apply_distractor_patch(
    image: Image.Image,
    source_patch_box: list,
    dest_box: list,
    seed: Optional[int] = None,
) -> Tuple[Image.Image, dict]:
    """
    Crop a patch from `source_patch_box` and paste it at `dest_box`.

    The source comes from a non-overlapping background region of the same image
    (computed by target_region_utils.find_distractor_source_patch). Pasting a
    same-image patch near a relevant object adds visually plausible but misleading
    evidence without introducing out-of-distribution textures.

    If the source and destination sizes differ, the patch is resized with bilinear
    interpolation. Image geometry is preserved → RSA is valid.

    Returns:
        (modified_image, metadata_dict)
    """
    W, H = image.size
    sx1, sy1, sx2, sy2 = _denorm(source_patch_box, W, H)
    dx1, dy1, dx2, dy2 = _denorm(dest_box, W, H)
    dx2, dy2 = min(dx2, W), min(dy2, H)

    if sx2 <= sx1 or sy2 <= sy1 or dx2 <= dx1 or dy2 <= dy1:
        return image.copy(), {
            "source_patch_box": source_patch_box,
            "dest_box":         dest_box,
            "skip_reason":      "zero-area source or destination box",
        }

    patch = image.crop((sx1, sy1, sx2, sy2)).convert("RGB")
    dw, dh = dx2 - dx1, dy2 - dy1
    if patch.size != (dw, dh):
        patch = patch.resize((dw, dh), Image.BILINEAR)

    img = image.copy().convert("RGB")
    img.paste(patch, (dx1, dy1))

    return img, {
        "source_patch_box": source_patch_box,
        "dest_box":         dest_box,
        "seed":             seed,
        "skip_reason":      None,
    }


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

def apply_targeted_perturbation(
    image: Image.Image,
    spec: PerturbationSpec,
    obj1_box: list,
    obj2_box: list,
    seed: Optional[int] = None,
) -> Tuple[Image.Image, dict]:
    """
    Apply the perturbation described by `spec` to `image`.

    Computes all required regions from obj1_box and obj2_box internally.
    Returns (perturbed_image, metadata_dict).

    Args:
        image:    clean PIL RGB image
        spec:     PerturbationSpec describing what to apply
        obj1_box: normalized [xmin, ymin, xmax, ymax] of object 1 (subj)
        obj2_box: normalized [xmin, ymin, xmax, ymax] of object 2 (obj)
        seed:     RNG seed for placement reproducibility
    """
    from target_region_utils import (
        get_union_box, get_relation_region, box_area,
        find_distractor_source_patch, distractor_destination_near_box,
    )

    if spec.perturbation_type == "occlusion":
        target = obj1_box if spec.target == "object_1" else obj2_box
        img, meta = apply_object_occlusion(image, target, spec.severity, seed=seed)
        meta["perturbation_target"] = spec.target
        return img, meta

    elif spec.perturbation_type == "mask":
        relation_box = get_relation_region(obj1_box, obj2_box, pad=0.05)
        img, meta = apply_relation_mask(image, relation_box, spec.severity, seed=seed)
        meta["perturbation_target"] = spec.target
        return img, meta

    elif spec.perturbation_type == "distractor":
        union_box = get_union_box(obj1_box, obj2_box)
        rel_box   = get_relation_region(obj1_box, obj2_box, pad=0.05)
        rel_area  = box_area(rel_box)

        # Patch size: severity fraction of relation-region area (square patch)
        ratio    = DISTRACTOR_SEVERITY_TO_RATIO[spec.severity]
        side     = max(0.02, min(0.4, (rel_area * ratio) ** 0.5))

        src_box = find_distractor_source_patch(
            avoid_boxes=[obj1_box, obj2_box, union_box],
            patch_w=side, patch_h=side, seed=seed,
        )
        if src_box is None:
            # Fallback: top-left corner of image (unlikely to overlap objects)
            src_box = [0.0, 0.0, side, side]

        near = obj1_box if spec.target == "near_obj1" else obj2_box
        dst_box = distractor_destination_near_box(near, side, side, seed=seed)

        img, meta = apply_distractor_patch(image, src_box, dst_box, seed=seed)
        meta["perturbation_target"] = spec.target
        meta["patch_size_norm"]     = side
        return img, meta

    else:
        raise ValueError(f"Unknown perturbation_type: {spec.perturbation_type!r}")
