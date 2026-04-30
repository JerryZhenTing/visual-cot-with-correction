"""
VSR dataset loader.

==== ASSUMPTIONS (all in one place) ====
1. Dataset source : HuggingFace 'cambridgeltl/vsr_zeroshot', split 'validation'
2. Image field    : 'image' — expected to be a PIL Image (or HF Image dict).
                    Fallback: 'image_link' URL field → download + cache to data/images/
3. Label field    : 'label' — ClassLabel with names ['false', 'true'] (0=false, 1=true).
                    Also handles raw int (0/1) or string ('false'/'true').
4. Bounding boxes : NOT available in the VSR GitHub annotation files.
                    The GitHub files only have: relation, subj, obj, label, image_link.
                    Bounding boxes would require COCO annotation lookup (not implemented).
                    RSA/IoU will remain N/A until COCO bboxes are integrated.
5. Annotation file: data/vsr_annotations.jsonl (JSONL format, one entry per line).
                    Provides: relation, subj, obj fields for per-relation analysis.
                    Auto-detected if the file exists.
6. Annotation matching: by caption string (must match exactly).
7. Target region  : always None for now (no bounding boxes available).
=========================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from PIL import Image

# ---- Dataset configuration (change these if your setup differs) ----
HF_DATASET_NAME = "cambridgeltl/vsr_zeroshot"
HF_SPLIT = "validation"

# Auto-detect annotation file. Prefer vsr_with_bboxes.jsonl (has COCO bboxes)
# over vsr_annotations.jsonl (relation/subj/obj only, no bboxes).
# Run scripts/download_annotations.sh then src/build_vsr_bboxes.py to get bboxes.
_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_ANN_WITH_BBOXES = os.path.join(_ROOT_DIR, "data", "vsr_with_bboxes.jsonl")
_ANN_BASIC       = os.path.join(_ROOT_DIR, "data", "vsr_annotations.jsonl")

if os.path.exists(_ANN_WITH_BBOXES):
    ANNOTATION_FILE: Optional[str] = _ANN_WITH_BBOXES
elif os.path.exists(_ANN_BASIC):
    ANNOTATION_FILE: Optional[str] = _ANN_BASIC
else:
    ANNOTATION_FILE: Optional[str] = None

IMAGE_CACHE_DIR: Optional[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "images"
)
DEV_SUBSET_SIZE: int = 20

# Bounding box format used in vsr_with_bboxes.jsonl (from COCO annotations).
# "coco" = [x, y, w, h] absolute pixels  →  _bbox_to_xyxy converts to xyxy.
# "xyxy" = [xmin, ymin, xmax, ymax]      →  passed through unchanged.
ANNOTATION_BBOX_FORMAT: str = "coco"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class VSRExample:
    example_id: str
    image: Image.Image          # PIL RGB image
    image_path: Optional[str]   # local path, if loaded from disk
    image_width: int
    image_height: int
    caption: str
    label: bool                 # True = statement is true
    relation: Optional[str]     # spatial relation type e.g. "to the left of"
    subj: Optional[str]         # subject object name e.g. "cat"
    obj: Optional[str]          # object name e.g. "dog"
    obj1_bbox: Optional[list]   # always None (COCO bboxes not yet integrated)
    obj2_bbox: Optional[list]   # always None

    def target_box_normalized(self) -> Optional[list]:
        """
        Return the union of obj1_bbox and obj2_bbox as a normalized
        [xmin, ymin, xmax, ymax] box.  Returns None if boxes are unavailable.
        Respects ANNOTATION_BBOX_FORMAT ("coco" or "xyxy").
        """
        if self.obj1_bbox is None or self.obj2_bbox is None:
            return None

        b1 = _bbox_to_xyxy(self.obj1_bbox)
        b2 = _bbox_to_xyxy(self.obj2_bbox)

        union_abs = [
            min(b1[0], b2[0]),
            min(b1[1], b2[1]),
            max(b1[2], b2[2]),
            max(b1[3], b2[3]),
        ]

        W, H = self.image_width, self.image_height
        # Clip to [0, 1] — COCO annotations near image edges can exceed image dimensions
        return [
            max(0.0, min(1.0, union_abs[0] / W)),
            max(0.0, min(1.0, union_abs[1] / H)),
            max(0.0, min(1.0, union_abs[2] / W)),
            max(0.0, min(1.0, union_abs[3] / H)),
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bbox_to_xyxy(b: list) -> list:
    """Convert a bbox to [xmin, ymin, xmax, ymax] based on ANNOTATION_BBOX_FORMAT."""
    if ANNOTATION_BBOX_FORMAT == "coco":
        x, y, w, h = b
        return [x, y, x + w, y + h]
    else:  # already xyxy
        return list(b)


def _extract_bboxes(ann: dict) -> tuple:
    """
    Extract obj1 and obj2 bboxes from an annotation dict.
    Tries multiple field name conventions in order.
    Returns (obj1_bbox, obj2_bbox), either may be None.
    """
    b1 = (ann.get("obj1_bbox") or ann.get("bbox_1") or
          ann.get("bbox1") or ann.get("subject_bbox"))
    b2 = (ann.get("obj2_bbox") or ann.get("bbox_2") or
          ann.get("bbox2") or ann.get("object_bbox"))
    return b1, b2


def _decode_label(raw) -> bool:
    """Convert raw label (int, str, or bool) to Python bool."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw)   # 0 → False, 1 → True
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return bool(raw)


def _load_pil_from_row(row: dict) -> Image.Image:
    """
    Extract a PIL Image from a HuggingFace dataset row.

    Handles three cases:
      1. row['image'] is already a PIL Image
      2. row['image'] is a dict {'bytes': ..., 'path': ...} (HF Image feature)
      3. row['image_link'] is a URL → download and cache to IMAGE_CACHE_DIR
    """
    if "image" in row:
        img_data = row["image"]

        # Direct PIL Image
        if hasattr(img_data, "convert"):
            return img_data.convert("RGB")

        # HF Image feature dict
        if isinstance(img_data, dict):
            if img_data.get("bytes"):
                return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
            if img_data.get("path"):
                return Image.open(img_data["path"]).convert("RGB")

    # Fallback: download from URL
    if "image_link" in row:
        return _download_image(row["image_link"])

    raise ValueError(
        f"Cannot find image in dataset row. Available keys: {list(row.keys())}"
    )


def _download_image(url: str) -> Image.Image:
    """Download an image from a URL, caching it locally."""
    import hashlib
    import urllib.request

    if IMAGE_CACHE_DIR:
        os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
        filename = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        cache_path = os.path.join(IMAGE_CACHE_DIR, filename)
        if os.path.exists(cache_path):
            return Image.open(cache_path).convert("RGB")
        urllib.request.urlretrieve(url, cache_path)
        return Image.open(cache_path).convert("RGB")

    with urllib.request.urlopen(url) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_vsr_dev_subset(
    n: int = DEV_SUBSET_SIZE,
    annotation_file: Optional[str] = ANNOTATION_FILE,
) -> list[VSRExample]:
    """
    Load the first `n` examples from the VSR validation split.

    Optionally merges bounding-box annotations from a local JSON file.
    Returns a list of VSRExample objects.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    print(f"Downloading/loading {HF_DATASET_NAME} ({HF_SPLIT} split)...")
    ds = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)

    # Build annotation lookup keyed by caption
    annotations: dict[str, dict] = {}
    if annotation_file and os.path.exists(annotation_file):
        with open(annotation_file) as f:
            if annotation_file.endswith(".jsonl"):
                ann_list = [json.loads(line) for line in f if line.strip()]
            else:
                ann_list = json.load(f)
        for ann in ann_list:
            annotations[ann["caption"]] = ann
        print(f"Loaded {len(annotations)} annotation entries from {annotation_file}")

    examples: list[VSRExample] = []
    for i, row in enumerate(ds):
        if i >= n:
            break

        try:
            img = _load_pil_from_row(row)
        except Exception as e:
            print(f"  Warning: could not load image for example {i}: {e}")
            continue

        W, H = img.size
        caption = row["caption"]
        label = _decode_label(row["label"])

        ann = annotations.get(caption, {})
        obj1_bbox, obj2_bbox = _extract_bboxes(ann)

        examples.append(VSRExample(
            example_id=f"vsr_{i:04d}",
            image=img,
            image_path=None,
            image_width=W,
            image_height=H,
            caption=caption,
            label=label,
            relation=ann.get("relation"),
            subj=ann.get("subj"),
            obj=ann.get("obj"),
            obj1_bbox=obj1_bbox,
            obj2_bbox=obj2_bbox,
        ))

    n_with_boxes = sum(1 for e in examples if e.obj1_bbox is not None and e.obj2_bbox is not None)
    print(f"Loaded {len(examples)} VSR examples  ({n_with_boxes} with bounding box annotations).")
    return examples


# ---------------------------------------------------------------------------
# Indexed loader (for matched subset evaluation)
# ---------------------------------------------------------------------------

def load_vsr_by_indices(
    indices: list[int],
    annotation_file: Optional[str] = ANNOTATION_FILE,
) -> list[VSRExample]:
    """
    Load specific VSR validation examples by their HF dataset indices.

    Use this with a saved subset file (from eval_config.create_subset) to
    guarantee that all models, methods, and conditions use the same examples.

    Args:
        indices:         list of integer HF dataset indices to load (sorted order)
        annotation_file: optional annotation file for bboxes and relation info
    Returns:
        list of VSRExample objects in the same order as `indices`
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    print(f"Loading {HF_DATASET_NAME} ({HF_SPLIT} split) for {len(indices)} indexed examples...")
    ds = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)

    # Build annotation lookup
    annotations: dict[str, dict] = {}
    if annotation_file and os.path.exists(annotation_file):
        with open(annotation_file) as f:
            ann_list = (
                [json.loads(line) for line in f if line.strip()]
                if annotation_file.endswith(".jsonl")
                else json.load(f)
            )
        for ann in ann_list:
            annotations[ann["caption"]] = ann
        print(f"Loaded {len(annotations)} annotation entries from {annotation_file}")

    index_set = set(indices)
    examples: list[VSRExample] = []
    # Map from index → position in `indices` so we preserve caller order
    idx_to_pos: dict[int, int] = {idx: pos for pos, idx in enumerate(indices)}
    slot: list[Optional[VSRExample]] = [None] * len(indices)

    for i, row in enumerate(ds):
        if i not in index_set:
            continue

        try:
            img = _load_pil_from_row(row)
        except Exception as e:
            print(f"  Warning: could not load image for HF index {i}: {e}")
            continue

        W, H = img.size
        caption = row["caption"]
        label = _decode_label(row["label"])
        ann = annotations.get(caption, {})
        obj1_bbox, obj2_bbox = _extract_bboxes(ann)

        ex = VSRExample(
            example_id=f"vsr_{i:04d}",
            image=img,
            image_path=None,
            image_width=W,
            image_height=H,
            caption=caption,
            label=label,
            relation=ann.get("relation"),
            subj=ann.get("subj"),
            obj=ann.get("obj"),
            obj1_bbox=obj1_bbox,
            obj2_bbox=obj2_bbox,
        )
        slot[idx_to_pos[i]] = ex

        if len([s for s in slot if s is not None]) == len(indices):
            break  # found all requested indices

    examples = [s for s in slot if s is not None]
    n_with_boxes = sum(1 for e in examples if e.obj1_bbox is not None and e.obj2_bbox is not None)
    print(f"Loaded {len(examples)}/{len(indices)} indexed examples  "
          f"({n_with_boxes} with bounding box annotations).")
    return examples


# ---------------------------------------------------------------------------
# Inspection helper
# ---------------------------------------------------------------------------

def print_example(ex: VSRExample) -> None:
    """Pretty-print one VSRExample for manual inspection."""
    sep = "=" * 60
    print(sep)
    print(f"ID           : {ex.example_id}")
    print(f"Caption      : {ex.caption}")
    print(f"Label        : {ex.label}")
    print(f"Image size   : {ex.image_width} x {ex.image_height}  (mode: {ex.image.mode})")
    print(f"Relation     : {ex.relation}")
    print(f"Subj / Obj   : {ex.subj} / {ex.obj}")
    print(f"obj1_bbox    : {ex.obj1_bbox}  (N/A — COCO bboxes not integrated)")
    print(f"obj2_bbox    : {ex.obj2_bbox}")
    print(sep)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = load_vsr_dev_subset(n=3)
    for ex in examples:
        print_example(ex)
