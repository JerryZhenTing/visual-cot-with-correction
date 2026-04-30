"""
VSR Guidance Dataset for the learned visual guidance policy.

Loads VSR examples that have valid bounding box annotations and builds
train / validation / test splits for supervised and RL training.

Target relation box = padded union of obj1 + obj2 boxes (normalized xyxy).
Examples without valid boxes for both objects are skipped.

Usage:
    ds = VSRGuidanceDataset.load_or_build(
        subset_file="data/subsets/vsr_n200_seq.json",
        cache_path="data/vsr_guidance.json",
        split="train",
        seed=42,
    )
    item = ds[0]
    # item keys: example_id, image, image_path, caption, answer,
    #            obj1_box, obj2_box, target_box, relation, subj, obj
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Optional

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_config import load_subset_indices
from load_vsr import ANNOTATION_FILE, load_vsr_by_indices, load_vsr_dev_subset
from target_region_utils import (
    get_boxes_for_example,
    get_relation_region,
)

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Default split fractions
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# Test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15


@dataclass
class GuidanceExample:
    example_id:  str
    image_path:  Optional[str]
    caption:     str
    answer:      str             # "true" or "false"
    obj1_box:    list            # normalized xyxy
    obj2_box:    list            # normalized xyxy
    target_box:  list            # padded union, normalized xyxy
    relation:    Optional[str]
    subj:        Optional[str]
    obj:         Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GuidanceExample":
        return cls(**d)


class VSRGuidanceDataset:
    """
    Iterable dataset of GuidanceExample objects for one split.

    Each __getitem__ returns a dict with:
      image       : PIL.Image (RGB)
      example_id  : str
      image_path  : str or None
      caption     : str
      answer      : "true" | "false"
      obj1_box    : [xmin,ymin,xmax,ymax]
      obj2_box    : [xmin,ymin,xmax,ymax]
      target_box  : [xmin,ymin,xmax,ymax]
      relation    : str or None
      subj        : str or None
      obj         : str or None
    """

    def __init__(self, examples: list[GuidanceExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        image = _load_image(ex.image_path)
        return {
            "image":      image,
            "example_id": ex.example_id,
            "image_path": ex.image_path,
            "caption":    ex.caption,
            "answer":     ex.answer,
            "obj1_box":   ex.obj1_box,
            "obj2_box":   ex.obj2_box,
            "target_box": ex.target_box,
            "relation":   ex.relation,
            "subj":       ex.subj,
            "obj":        ex.obj,
        }

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def load_or_build(
        cls,
        subset_file: Optional[str] = None,
        cache_path: str = "data/vsr_guidance.json",
        split: str = "train",
        seed: int = 42,
        pad: float = 0.05,
        n_fallback: int = 200,
    ) -> "VSRGuidanceDataset":
        """
        Load from cache if available, otherwise build from VSR annotations.

        Args:
            subset_file : path to subset JSON (e.g. data/subsets/vsr_n200_seq.json)
            cache_path  : where to cache/load processed examples
            split       : "train" | "val" | "test" | "all"
            seed        : random seed for deterministic split
            pad         : padding for relation region (default 0.05)
            n_fallback  : number of examples if no subset_file
        """
        cache_abs = os.path.join(_ROOT, cache_path) if not os.path.isabs(cache_path) else cache_path

        if os.path.exists(cache_abs):
            print(f"Loading guidance dataset from cache: {cache_abs}")
            with open(cache_abs) as f:
                data = json.load(f)
            all_examples = [GuidanceExample.from_dict(d) for d in data["examples"]]
            print(f"  {len(all_examples)} examples loaded.")
        else:
            all_examples = _build_examples(subset_file, n_fallback, pad)
            _save_cache(all_examples, cache_abs)

        splits = _make_splits(all_examples, seed)

        if split == "all":
            return cls(all_examples)
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}, choose from train/val/test/all")
        return cls(splits[split])

    @classmethod
    def from_cache(cls, cache_path: str, split: str = "train", seed: int = 42) -> "VSRGuidanceDataset":
        """Load directly from a saved cache file."""
        with open(cache_path) as f:
            data = json.load(f)
        all_examples = [GuidanceExample.from_dict(d) for d in data["examples"]]
        splits = _make_splits(all_examples, seed)
        if split == "all":
            return cls(all_examples)
        return cls(splits[split])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_examples(
    subset_file: Optional[str],
    n_fallback: int,
    pad: float,
) -> list[GuidanceExample]:
    """Load VSR data, filter to examples with valid boxes, build GuidanceExamples."""
    if subset_file:
        path = subset_file if os.path.isabs(subset_file) else os.path.join(_ROOT, subset_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Subset file not found: {path}")
        indices = load_subset_indices(path)
        print(f"Loading {len(indices)} VSR examples from {path}...")
        vsr_examples = load_vsr_by_indices(indices, annotation_file=ANNOTATION_FILE)
    else:
        print(f"No subset file — loading first {n_fallback} examples.")
        vsr_examples = load_vsr_dev_subset(n=n_fallback, annotation_file=ANNOTATION_FILE)

    examples: list[GuidanceExample] = []
    n_skipped = 0
    skip_reasons: dict[str, int] = {}

    for ex in vsr_examples:
        obj1_box, obj2_box, skip_reason = get_boxes_for_example(ex)
        if skip_reason:
            n_skipped += 1
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            continue

        target_box = get_relation_region(obj1_box, obj2_box, pad=pad)

        examples.append(GuidanceExample(
            example_id = ex.example_id,
            image_path = ex.image_path,
            caption    = ex.caption,
            answer     = "true" if ex.label else "false",
            obj1_box   = obj1_box,
            obj2_box   = obj2_box,
            target_box = target_box,
            relation   = ex.relation,
            subj       = ex.subj,
            obj        = ex.obj,
        ))

    print(f"\nGuidance dataset built:")
    print(f"  Total VSR examples   : {len(vsr_examples)}")
    print(f"  With valid boxes     : {len(examples)}")
    print(f"  Skipped              : {n_skipped}")
    for reason, count in skip_reasons.items():
        print(f"    {reason}: {count}")

    return examples


def _save_cache(examples: list[GuidanceExample], cache_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    data = {"n": len(examples), "examples": [e.to_dict() for e in examples]}
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved guidance dataset cache → {cache_path}")


def _make_splits(
    examples: list[GuidanceExample],
    seed: int,
) -> dict[str, list[GuidanceExample]]:
    """Deterministic 70/15/15 split by seed."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def _load_image(image_path: Optional[str]) -> Optional[Image.Image]:
    """Load PIL image from path. Returns None if path is missing."""
    if image_path is None or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: could not load {image_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI: build and cache the dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build and cache the VSR guidance dataset.")
    p.add_argument("--subset",     default=None,
                   help="Subset JSON (omit to use entire VSR validation set)")
    p.add_argument("--cache",      default="data/vsr_guidance_full.json")
    p.add_argument("--n",          type=int, default=1222,
                   help="Number of examples to load when no subset file given (default: full validation set)")
    p.add_argument("--pad",        type=float, default=0.05)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    ds = VSRGuidanceDataset.load_or_build(
        subset_file=args.subset,
        cache_path=args.cache,
        split="all",
        seed=args.seed,
        pad=args.pad,
        n_fallback=args.n,
    )
    print(f"\nTotal examples: {len(ds)}")
    splits = _make_splits(ds.examples, args.seed)
    for name, exs in splits.items():
        print(f"  {name}: {len(exs)}")
