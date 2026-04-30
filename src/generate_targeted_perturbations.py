"""
Generate targeted perturbation images for the VSR evidence-disruption evaluation.

Loads the matched VSR subset, applies all 15 targeted perturbation specs to each
example that has valid object bounding boxes, and saves results to data/targeted/.

Examples without box annotations are skipped (recorded in metadata with skip_reason).
No GPU is needed — this script is pure PIL/numpy image processing.

Output structure:
  data/targeted/
    occlusion/object_1/low/<example_id>.jpg
    occlusion/object_1/medium/<example_id>.jpg
    occlusion/object_1/high/<example_id>.jpg
    occlusion/object_2/{low,medium,high}/<example_id>.jpg
    mask/union/{low,medium,high}/<example_id>.jpg
    distractor/near_obj1/{low,medium,high}/<example_id>.jpg
    distractor/near_obj2/{low,medium,high}/<example_id>.jpg
    metadata.json

metadata.json index:
  {
    "subset_file": ...,
    "n_loaded": ...,
    "n_with_boxes": ...,
    "specs": [...],
    "created_at": ...,
    "examples": [
      {
        "example_id": ...,
        "has_boxes": true/false,
        "skip_reason": null or "...",
        "obj1_box": [...],      # normalized xyxy, only when has_boxes
        "obj2_box": [...],
        "union_box": [...],
        "relation_region": [...],
        "perturbations": {
          "<display_name>": {
            "image_path": "data/targeted/<subdir>/<id>.jpg",
            "rsa_valid": true,
            "seed": ...,
            "skip_reason": null or "...",
            ...metadata from apply_targeted_perturbation...
          }
        }
      }
    ]
  }

Usage:
  python src/generate_targeted_perturbations.py \\
      --subset data/subsets/vsr_n200_seq.json \\
      --out-dir data/targeted \\
      --seed 42

  # Dry run (no images written):
  python src/generate_targeted_perturbations.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adversarial_perturbations import (
    PerturbationSpec,
    apply_targeted_perturbation,
    get_all_targeted_perturbation_specs,
)
from eval_config import load_subset_indices
from load_vsr import ANNOTATION_FILE, load_vsr_by_indices, load_vsr_dev_subset
from target_region_utils import (
    get_boxes_for_example,
    get_relation_region,
    get_union_box,
)

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _example_seed(base_seed: int, example_id: str, spec_idx: int) -> int:
    """Deterministic per-(example, spec) seed."""
    id_hash = abs(hash(example_id)) % 100_000
    return base_seed * 1_000_000 + id_hash * 100 + spec_idx


def generate(
    subset_file: str | None,
    out_dir: str,
    base_seed: int,
    n_fallback: int,
    dry_run: bool,
    jpeg_quality: int,
) -> dict:
    """
    Generate all targeted perturbation images for the given subset.

    Returns the metadata dict (also written to out_dir/metadata.json).
    """
    # Load examples
    if subset_file:
        path = subset_file if os.path.isabs(subset_file) else os.path.join(_ROOT, subset_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Subset file not found: {path}")
        indices = load_subset_indices(path)
        print(f"Loading {len(indices)} indexed examples from {path}...")
        examples = load_vsr_by_indices(indices, annotation_file=ANNOTATION_FILE)
    else:
        print(f"No subset file — loading first {n_fallback} examples.")
        examples = load_vsr_dev_subset(n=n_fallback, annotation_file=ANNOTATION_FILE)
        subset_file = f"first-{n_fallback} (no subset file)"

    specs = get_all_targeted_perturbation_specs()
    spec_display_names = [s.display_name for s in specs]

    n_with_boxes = 0
    n_skipped    = 0
    n_images_written = 0

    example_records = []

    for ex in examples:
        obj1_box, obj2_box, skip_reason = get_boxes_for_example(ex)

        if skip_reason:
            n_skipped += 1
            example_records.append({
                "example_id":  ex.example_id,
                "has_boxes":   False,
                "skip_reason": skip_reason,
            })
            continue

        n_with_boxes += 1
        union_box    = get_union_box(obj1_box, obj2_box)
        rel_region   = get_relation_region(obj1_box, obj2_box, pad=0.05)
        perturbations: dict = {}

        for idx, spec in enumerate(specs):
            seed = _example_seed(base_seed, ex.example_id, idx)

            try:
                perturbed_img, pert_meta = apply_targeted_perturbation(
                    ex.image, spec, obj1_box, obj2_box, seed=seed,
                )
            except Exception as exc:
                perturbations[spec.display_name] = {
                    "image_path": None,
                    "rsa_valid":  spec.rsa_valid,
                    "seed":       seed,
                    "skip_reason": f"perturbation failed: {exc}",
                }
                continue

            if pert_meta.get("skip_reason"):
                perturbations[spec.display_name] = {
                    "image_path":  None,
                    "rsa_valid":   spec.rsa_valid,
                    "seed":        seed,
                    **pert_meta,
                }
                continue

            # Build output path
            subdir_abs = os.path.join(_ROOT if not os.path.isabs(out_dir) else "", out_dir, spec.subdir)
            img_path_abs = os.path.join(subdir_abs, f"{ex.example_id}.jpg")
            # Store relative path in metadata for portability
            img_path_rel = os.path.join(out_dir, spec.subdir, f"{ex.example_id}.jpg")

            if not dry_run:
                os.makedirs(subdir_abs, exist_ok=True)
                perturbed_img.save(img_path_abs, format="JPEG", quality=jpeg_quality)
                n_images_written += 1

            perturbations[spec.display_name] = {
                "image_path": img_path_rel,
                "rsa_valid":  spec.rsa_valid,
                "seed":       seed,
                **pert_meta,
            }

        example_records.append({
            "example_id":    ex.example_id,
            "caption":       ex.caption,
            "has_boxes":     True,
            "skip_reason":   None,
            "obj1_box":      obj1_box,
            "obj2_box":      obj2_box,
            "union_box":     union_box,
            "relation_region": rel_region,
            "perturbations": perturbations,
        })

        if n_with_boxes % 20 == 0:
            print(f"  {n_with_boxes} examples processed...")

    # Build and save metadata
    meta = {
        "subset_file":    subset_file,
        "n_loaded":       len(examples),
        "n_with_boxes":   n_with_boxes,
        "n_skipped":      n_skipped,
        "n_images_written": n_images_written,
        "specs": [
            {"display_name": s.display_name, "perturbation_type": s.perturbation_type,
             "target": s.target, "severity": s.severity, "subdir": s.subdir}
            for s in specs
        ],
        "created_at": datetime.datetime.now().isoformat(),
        "dry_run":    dry_run,
        "examples":   example_records,
    }

    meta_path = os.path.join(out_dir if os.path.isabs(out_dir) else os.path.join(_ROOT, out_dir),
                             "metadata.json")
    if not dry_run:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Metadata → {meta_path}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Examples loaded:     {len(examples)}")
    print(f"  With valid boxes:    {n_with_boxes}")
    print(f"  Skipped (no boxes):  {n_skipped}")
    print(f"  Perturbation specs:  {len(specs)}")
    print(f"  Images written:      {n_images_written}  (expected {n_with_boxes * len(specs)})")

    return meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate targeted perturbation images for VSR evaluation.")
    p.add_argument("--subset",   default="data/subsets/vsr_n200_seq.json",
                   help="Subset JSON from eval_config.py (default: data/subsets/vsr_n200_seq.json)")
    p.add_argument("--out-dir",  default="data/targeted",
                   help="Output directory (default: data/targeted)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Base RNG seed for all perturbations (default: 42)")
    p.add_argument("--n",        type=int, default=200,
                   help="Number of examples if no subset file (default: 200)")
    p.add_argument("--quality",  type=int, default=92,
                   help="JPEG save quality 1-100 (default: 92)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Run without writing any files (for testing)")
    args = p.parse_args()

    generate(
        subset_file  = args.subset,
        out_dir      = args.out_dir,
        base_seed    = args.seed,
        n_fallback   = args.n,
        dry_run      = args.dry_run,
        jpeg_quality = args.quality,
    )


if __name__ == "__main__":
    main()
