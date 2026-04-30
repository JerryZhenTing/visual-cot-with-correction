"""
Post-process existing result files to fix Qwen2.5-VL pixel-coordinate boxes.

Qwen2.5-VL outputs bounding boxes in [0, 1000] coordinates rather than the
[0, 1] range requested in the prompts.  All previously saved results have
box_valid=False because of this.

This script:
  1. Walks all result files (clean + corrupted, all methods)
  2. For each result, normalizes any box with values > 1 by dividing by 1000
  3. Revalidates the box (xmin<xmax, ymin<ymax, all in [0,1])
  4. Recomputes iou and rsa against target_box
  5. For clean visual_cot (where target_box was missing), backfills target_box
     from data/vsr_with_bboxes.jsonl matched by caption
  6. Overwrites the result file in-place (original backed up as .bak)

Usage:
    cd /path/to/project
    python src/fix_boxes.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import compute_rsa, iou as compute_iou
from bqs import compute_bqs

_ROOT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RSA_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Box normalization
# ---------------------------------------------------------------------------

def _normalize_box(box):
    """Divide by 1000 if any value > 1.0; return None if box is None."""
    if box is None:
        return None
    if any(v > 1.0 for v in box):
        return [v / 1000.0 for v in box]
    return box


def _clip_target(box):
    """Clip target_box to [0,1] — COCO annotations near image edges can exceed bounds."""
    if box is None:
        return None
    return [max(0.0, min(1.0, v)) for v in box]


def _validate_box(box):
    """Return (valid, invalid_reason) for a normalized box."""
    if box is None:
        return False, "box is None"
    if not all(0.0 <= v <= 1.0 for v in box):
        return False, f"values outside [0,1] after normalization: {box}"
    xmin, ymin, xmax, ymax = box
    if xmin >= xmax:
        return False, f"xmin ({xmin:.4f}) >= xmax ({xmax:.4f})"
    if ymin >= ymax:
        return False, f"ymin ({ymin:.4f}) >= ymax ({ymax:.4f})"
    return True, None


def _spatial(box, box_valid, target_box):
    """Return (iou, rsa) or (None, None)."""
    if box_valid and box and target_box:
        iou_val = compute_iou(box, target_box)
        rsa_val = compute_rsa(box, target_box, RSA_THRESHOLD)
        return iou_val, rsa_val
    return None, None


# ---------------------------------------------------------------------------
# Target box backfill (for clean visual_cot which was run before bbox file)
# ---------------------------------------------------------------------------

def _load_caption_to_target(jsonl_path: str) -> dict:
    """
    Build caption → (obj1_bbox, obj2_bbox, image_width, image_height) mapping
    from vsr_with_bboxes.jsonl.

    Returns caption → target_box_normalized (or None if bboxes missing).
    """
    if not os.path.exists(jsonl_path):
        return {}

    mapping = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            caption = ann.get("caption")
            if not caption:
                continue

            b1 = (ann.get("obj1_bbox") or ann.get("bbox_1") or ann.get("bbox1"))
            b2 = (ann.get("obj2_bbox") or ann.get("bbox_2") or ann.get("bbox2"))
            W  = ann.get("image_width")
            H  = ann.get("image_height")

            if b1 and b2 and W and H:
                # Convert COCO [x,y,w,h] to xyxy
                def to_xyxy(b):
                    x, y, w, h = b
                    return [x, y, x + w, y + h]

                b1x = to_xyxy(b1)
                b2x = to_xyxy(b2)
                union = [
                    min(b1x[0], b2x[0]),
                    min(b1x[1], b2x[1]),
                    max(b1x[2], b2x[2]),
                    max(b1x[3], b2x[3]),
                ]
                # Clip to [0, 1] — COCO annotations near image edges can exceed bounds
                target = [
                    max(0.0, min(1.0, union[0]/W)),
                    max(0.0, min(1.0, union[1]/H)),
                    max(0.0, min(1.0, union[2]/W)),
                    max(0.0, min(1.0, union[3]/H)),
                ]
                mapping[caption] = target
            else:
                mapping[caption] = None

    return mapping


# ---------------------------------------------------------------------------
# Per-method fixers
# ---------------------------------------------------------------------------

def _fix_visual(result: dict, caption_to_target: dict) -> dict:
    r = dict(result)

    # Backfill target_box if missing
    if r.get("target_box") is None:
        r["target_box"] = caption_to_target.get(r.get("caption"))

    r["target_box"] = _clip_target(r.get("target_box"))
    target = r.get("target_box")
    box = _normalize_box(r.get("parsed_box"))
    valid, reason = _validate_box(box)
    r["parsed_box"]         = box
    r["box_valid"]          = valid
    r["box_invalid_reason"] = reason
    r["iou"], r["rsa"]      = _spatial(box, valid, target)
    return r


def _fix_verification(result: dict, caption_to_target: dict) -> dict:
    r = dict(result)

    if r.get("target_box") is None:
        r["target_box"] = caption_to_target.get(r.get("caption"))

    r["target_box"] = _clip_target(r.get("target_box"))
    target = r.get("target_box")

    for prefix in ("pass1", "pass2"):
        box_key    = f"{prefix}_box"
        valid_key  = f"{prefix}_box_valid"
        reason_key = f"{prefix}_box_invalid_reason"
        iou_key    = f"{prefix}_iou"
        rsa_key    = f"{prefix}_rsa"

        box = _normalize_box(r.get(box_key))
        valid, reason = _validate_box(box)
        r[box_key]   = box
        r[valid_key] = valid
        r[reason_key] = reason
        r[iou_key], r[rsa_key] = _spatial(box, valid, target)

    return r


def _fix_multistage(result: dict, caption_to_target: dict) -> dict:
    r = dict(result)

    if r.get("target_box") is None:
        r["target_box"] = caption_to_target.get(r.get("caption"))

    r["target_box"] = _clip_target(r.get("target_box"))
    target = r.get("target_box")

    for prefix, iou_k, rsa_k in [
        ("initial", "initial_iou", "initial_rsa"),
        ("revised", "revised_iou", "revised_rsa"),
    ]:
        box = _normalize_box(r.get(f"{prefix}_box"))
        valid, reason = _validate_box(box)
        r[f"{prefix}_box"]                = box
        r[f"{prefix}_box_valid"]          = valid
        r[f"{prefix}_box_invalid_reason"] = reason
        r[iou_k], r[rsa_k]               = _spatial(box, valid, target)

    # Recompute BQS with now-valid initial box
    stability = r.get("stability_score", 1.0)
    bqs_result = compute_bqs(
        box_valid      = r.get("initial_box_valid", False),
        box            = r.get("initial_box"),
        target_box     = target,
        reasoning      = r.get("initial_reasoning") or "",
        subj           = r.get("subj") or "",
        obj            = r.get("obj") or "",
        stability_score= stability if stability is not None else 1.0,
    )
    r["bqs"]              = bqs_result["bqs"]
    r["format_score"]     = bqs_result["format_score"]
    r["overlap_score"]    = bqs_result["overlap_score"]
    r["mention_score"]    = bqs_result["mention_score"]
    r["stability_score"]  = bqs_result["stability_score"]

    return r


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def _method_of(path: str) -> str:
    """Infer method from file path."""
    parts = path.replace("\\", "/").split("/")
    name  = os.path.basename(path).replace(".json", "")
    # Clean result files
    if "textual_cot_results" in name:
        return "textual"
    if "visual_cot_results" in name:
        return "visual"
    if "visual_cot_verification" in name:
        return "verification"
    if "multistage" in name:
        return "multistage"
    # Corrupted: path is results/corrupted/<method>/<condition>.json
    for m in ("textual", "visual", "verification", "multistage"):
        if f"/{m}/" in path.replace("\\", "/"):
            return m
    return "unknown"


def _fix_file(path: str, caption_to_target: dict, dry_run: bool = False) -> int:
    """
    Fix one result file.  Returns number of boxes fixed.
    """
    with open(path) as f:
        results = json.load(f)

    method  = _method_of(path)
    fixed   = 0
    changed = 0
    updated = []

    for r in results:
        if method == "textual":
            updated.append(r)          # no boxes to fix
            continue

        orig_valid = (
            r.get("box_valid") or
            r.get("pass1_box_valid") or
            r.get("initial_box_valid")
        )
        orig_iou = r.get("iou") or r.get("pass2_iou") or r.get("initial_iou")
        orig_bqs = r.get("bqs")
        orig_target = r.get("target_box")

        if method == "visual":
            r = _fix_visual(r, caption_to_target)
        elif method == "verification":
            r = _fix_verification(r, caption_to_target)
        elif method == "multistage":
            r = _fix_multistage(r, caption_to_target)

        new_valid = (
            r.get("box_valid") or
            r.get("pass1_box_valid") or
            r.get("initial_box_valid")
        )
        new_iou = r.get("iou") or r.get("pass2_iou") or r.get("initial_iou")
        new_bqs = r.get("bqs")
        new_target = r.get("target_box")

        if not orig_valid and new_valid:
            fixed += 1
        if orig_iou != new_iou or orig_bqs != new_bqs or orig_target != new_target:
            changed += 1

        updated.append(r)

    total_changes = fixed + changed
    if not dry_run and total_changes > 0:
        bak = path + ".bak"
        if not os.path.exists(bak):   # keep the original backup, not the already-fixed version
            shutil.copy2(path, bak)
        with open(path, "w") as f:
            json.dump(updated, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    return total_changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    ann_path         = os.path.join(_ROOT, "data", "vsr_with_bboxes.jsonl")
    caption_to_target = _load_caption_to_target(ann_path)
    print(f"Loaded {len(caption_to_target)} caption→target_box entries from {ann_path}\n")

    # Collect all result files
    result_files = []
    results_dir  = os.path.join(_ROOT, "results")

    for fname in os.listdir(results_dir):
        if fname.endswith(".json") and not fname.endswith(".bak"):
            result_files.append(os.path.join(results_dir, fname))

    corrupted_dir = os.path.join(results_dir, "corrupted")
    if os.path.exists(corrupted_dir):
        for method in os.listdir(corrupted_dir):
            method_dir = os.path.join(corrupted_dir, method)
            if os.path.isdir(method_dir):
                for fname in os.listdir(method_dir):
                    if fname.endswith(".json") and not fname.endswith(".bak"):
                        result_files.append(os.path.join(method_dir, fname))

    result_files.sort()
    total_fixed = 0

    for path in result_files:
        rel = os.path.relpath(path, _ROOT)
        n   = _fix_file(path, caption_to_target, dry_run=dry_run)
        status = f"updated {n} boxes" if n > 0 else "no change"
        print(f"  {'[DRY]' if dry_run else '[OK] '} {rel:<60} {status}")
        total_fixed += n

    print(f"\nTotal boxes fixed: {total_fixed}")
    if not dry_run and total_fixed > 0:
        print("Originals backed up as .bak files.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = p.parse_args()
    main(dry_run=args.dry_run)
