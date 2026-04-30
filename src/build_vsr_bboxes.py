"""
Enrich VSR annotations with COCO bounding boxes.

Reads  : data/vsr_annotations.jsonl
Reads  : data/coco_annotations/instances_train2017.json
Reads  : data/coco_annotations/instances_val2017.json
Writes : data/vsr_with_bboxes.jsonl

Each output entry = original VSR fields + obj1_bbox + obj2_bbox
(in COCO [x, y, w, h] format, absolute pixels).

Entries where both bboxes are found are counted as "covered".
Entries with missing bboxes still get written (with null for missing bbox)
so the full dataset is preserved.

Run on the login node (no GPU needed):
    cd /jet/home/zliu51/project
    python src/build_vsr_bboxes.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coco_bbox_lookup import COCOBBoxLookup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

VSR_ANN_PATH    = os.path.join(_ROOT, "data", "vsr_annotations.jsonl")
COCO_TRAIN_PATH = os.path.join(_ROOT, "data", "coco_annotations", "instances_train2017.json")
COCO_VAL_PATH   = os.path.join(_ROOT, "data", "coco_annotations", "instances_val2017.json")
OUTPUT_PATH     = os.path.join(_ROOT, "data", "vsr_with_bboxes.jsonl")


def _image_filename(entry: dict) -> str:
    """Extract just the filename from an image_link URL or image field."""
    # Prefer the 'image' field (just filename); fall back to last segment of image_link
    if entry.get("image"):
        return os.path.basename(entry["image"])
    link = entry.get("image_link", "")
    return os.path.basename(link)


def build_vsr_bboxes() -> None:
    # ---- Load VSR annotations ----
    if not os.path.exists(VSR_ANN_PATH):
        raise FileNotFoundError(
            f"VSR annotation file not found: {VSR_ANN_PATH}\n"
            "Run: bash scripts/download_annotations.sh"
        )

    with open(VSR_ANN_PATH) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(entries)} VSR annotation entries.")

    # ---- Build COCO bbox index ----
    print("Building COCO bbox index (may take ~30 sec for large files)...")
    lookup = COCOBBoxLookup(COCO_TRAIN_PATH, COCO_VAL_PATH)

    # ---- Coverage report ----
    cov = lookup.coverage(entries)
    print(f"\nCoverage report:")
    print(f"  Both subj+obj found : {cov['both_found']} / {cov['total']}  ({cov['coverage_rate']:.1%})")
    print(f"  Subj only           : {cov['subj_only']}")
    print(f"  Obj only            : {cov['obj_only']}")
    print(f"  Neither found       : {cov['neither']}")

    # ---- Enrich and write ----
    enriched = []
    for entry in entries:
        filename = _image_filename(entry)
        subj = entry.get("subj", "")
        obj  = entry.get("obj", "")

        obj1_bbox = lookup.get_bbox(filename, subj)
        obj2_bbox = lookup.get_bbox(filename, obj)

        out = dict(entry)
        out["obj1_bbox"] = obj1_bbox   # [x, y, w, h] or null
        out["obj2_bbox"] = obj2_bbox
        enriched.append(out)

    with open(OUTPUT_PATH, "w") as f:
        for entry in enriched:
            f.write(json.dumps(entry) + "\n")

    n_complete = sum(1 for e in enriched if e["obj1_bbox"] and e["obj2_bbox"])
    print(f"\nWrote {len(enriched)} entries to {OUTPUT_PATH}")
    print(f"  {n_complete} entries have both bboxes (RSA/IoU computable).")
    print(f"  {len(enriched) - n_complete} entries have missing bboxes (RSA will be N/A).")


if __name__ == "__main__":
    build_vsr_bboxes()
