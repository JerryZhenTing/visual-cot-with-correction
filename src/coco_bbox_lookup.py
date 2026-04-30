"""
COCO bounding box lookup for VSR examples.

Builds an in-memory index from (image_filename, category_name) to a list
of COCO bounding boxes, then provides a get_bbox() method that returns the
best (largest-area) bbox for a given image and object name.

Bounding boxes are returned in COCO format: [x, y, width, height]
(absolute pixel coordinates, top-left origin).

Usage:
    lookup = COCOBBoxLookup(
        train_path="data/coco_annotations/instances_train2017.json",
        val_path="data/coco_annotations/instances_val2017.json",
    )
    bbox = lookup.get_bbox("000000372029.jpg", "dog")
    # Returns [x, y, w, h] or None if not found
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional


class COCOBBoxLookup:
    """
    Index COCO instance annotations for fast bbox lookup by
    (image_filename, category_name).
    """

    def __init__(self, train_path: str, val_path: str):
        """
        Args:
            train_path: path to instances_train2017.json
            val_path:   path to instances_val2017.json
        """
        # index[filename][category_name] = list of [x, y, w, h]
        self._index: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for path in [train_path, val_path]:
            if not os.path.exists(path):
                print(f"  Warning: COCO annotation file not found: {path}")
                continue
            print(f"  Loading {os.path.basename(path)} ...")
            self._load(path)

        print(f"  Indexed {len(self._index)} images.")

    def _load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)

        # Build id → filename map
        id_to_filename: dict[int, str] = {
            img["id"]: img["file_name"] for img in data["images"]
        }

        # Build category_id → name map (lowercase for matching)
        id_to_cat: dict[int, str] = {
            cat["id"]: cat["name"].lower() for cat in data["categories"]
        }

        for ann in data["annotations"]:
            filename = id_to_filename.get(ann["image_id"])
            cat_name = id_to_cat.get(ann["category_id"])
            if filename and cat_name and ann.get("bbox"):
                self._index[filename][cat_name].append(ann["bbox"])

    def get_bbox(
        self,
        image_filename: str,
        category_name: str,
    ) -> Optional[list]:
        """
        Return the largest-area bounding box for the given image and category.

        Args:
            image_filename: COCO filename e.g. "000000372029.jpg"
            category_name:  object name e.g. "dog" (case-insensitive)
        Returns:
            [x, y, w, h] in absolute pixels, or None if not found
        """
        cat = category_name.lower().strip()
        bboxes = self._index.get(image_filename, {}).get(cat, [])

        if not bboxes:
            # Try partial match (e.g. "cell phone" → "phone")
            for key, boxes in self._index.get(image_filename, {}).items():
                if cat in key or key in cat:
                    bboxes = boxes
                    break

        if not bboxes:
            return None

        # Return the largest bbox by area
        return max(bboxes, key=lambda b: b[2] * b[3])

    def coverage(self, entries: list[dict]) -> dict:
        """
        Report how many VSR entries have both subj and obj bboxes found.

        Args:
            entries: list of VSR annotation dicts with 'image', 'subj', 'obj' fields
        Returns:
            dict with counts and fraction
        """
        total = len(entries)
        both = 0
        subj_only = 0
        obj_only = 0
        neither = 0

        for e in entries:
            filename = e.get("image", "")
            has_subj = self.get_bbox(filename, e.get("subj", "")) is not None
            has_obj = self.get_bbox(filename, e.get("obj", "")) is not None
            if has_subj and has_obj:
                both += 1
            elif has_subj:
                subj_only += 1
            elif has_obj:
                obj_only += 1
            else:
                neither += 1

        return {
            "total": total,
            "both_found": both,
            "subj_only": subj_only,
            "obj_only": obj_only,
            "neither": neither,
            "coverage_rate": both / total if total else 0.0,
        }
