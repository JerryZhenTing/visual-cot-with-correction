"""
Evaluation configuration for the matched VSR evaluation pipeline.

EvalConfig is the single source of truth for one evaluation run.
It specifies which model, method, and condition to run, which matched
subset to use, and where to write results.

Subset management functions (create_subset / load_subset_indices) handle
sampling a fixed random subset from the VSR validation split so that
all models, methods, and conditions run on exactly the same examples.

Quick start:
    # Create a 100-example matched subset (do this once):
    from eval_config import create_subset
    create_subset(n=100, seed=42, save_path="data/subsets/vsr_n100_s42.json")

    # Build a config and run:
    from eval_config import EvalConfig
    cfg = EvalConfig(
        model_name="qwen",
        method="visual",
        condition="blur-1",
        subset_file="data/subsets/vsr_n100_s42.json",
    )
    cfg.save()   # writes configs/qwen_visual_blur-1.cfg.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

VALID_MODELS = ("qwen", "llava")
VALID_METHODS = ("textual", "visual", "verification", "multistage")
VALID_CONDITIONS = (
    "clean",
    "blur-1", "blur-3", "blur-5",
    "noise-001", "noise-005", "noise-010",
    "rot-15", "rot-45", "rot-90",
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """
    Complete specification for one evaluation run.

    Attributes:
        model_name:       "qwen" or "llava"
        method:           inference method
        condition:        data condition ("clean" or a corruption display name)
        dataset:          always "vsr" for now
        n_examples:       size of the matched subset (used only if subset_file is None)
        subset_seed:      RNG seed used when creating the subset
        subset_file:      path to saved subset JSON (relative to project root);
                          if None, falls back to sequential first-N loading
        raw_results_dir:  directory for per-example result files (relative to root)
        max_new_tokens:   model generation limit
    """
    model_name:      str
    method:          str
    condition:       str
    dataset:         str  = "vsr"
    n_examples:      int  = 100
    subset_seed:     int  = 42
    subset_file:     Optional[str] = None
    raw_results_dir: str  = "results/raw"
    max_new_tokens:  int  = 512

    def __post_init__(self) -> None:
        if self.model_name not in VALID_MODELS:
            raise ValueError(f"Unknown model {self.model_name!r}. Choose from: {VALID_MODELS}")
        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown method {self.method!r}. Choose from: {VALID_METHODS}")
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Unknown condition {self.condition!r}. Choose from: {VALID_CONDITIONS}"
            )

    # ---- Derived paths ----

    @property
    def output_filename(self) -> str:
        return f"{self.model_name}_{self.method}_{self.condition}.json"

    @property
    def output_path(self) -> str:
        """Absolute path to the per-example result file."""
        return os.path.join(_ROOT, self.raw_results_dir, self.output_filename)

    @property
    def config_filename(self) -> str:
        return f"{self.model_name}_{self.method}_{self.condition}.cfg.json"

    # ---- Serialization ----

    def save(self, path: Optional[str] = None) -> str:
        """Write config to JSON. Returns the path written."""
        if path is None:
            path = os.path.join(_ROOT, "configs", self.config_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "EvalConfig":
        """Read config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Subset management
# ---------------------------------------------------------------------------

VSR_VALIDATION_SIZE = 1222  # cambridgeltl/vsr_zeroshot validation split size


def create_subset(
    n: int,
    seed: int,
    save_path: str,
    total: int = VSR_VALIDATION_SIZE,
    sequential: bool = False,
) -> list[int]:
    """
    Sample n indices from the VSR validation split and save them.

    All models, methods, and conditions should use the same subset file to
    guarantee matched comparisons.  Can be run locally — no GPU needed.

    Args:
        n:           number of examples to sample
        seed:        RNG seed (used only when sequential=False)
        save_path:   where to write the subset JSON (relative to project root)
        total:       total size of the VSR validation split (default: 1222)
        sequential:  if True, use the first n indices (0..n-1) instead of random
                     sampling — safer when the HF dataset cache is only partially
                     populated (which is common on HPC systems with limited storage)
    Returns:
        list of HF dataset integer indices (sorted)
    """
    import random
    import datetime

    if not os.path.isabs(save_path):
        save_path = os.path.join(_ROOT, save_path)

    if sequential:
        indices = list(range(min(n, total)))
        seed_used = None
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), min(n, total)))
        seed_used = seed

    meta = {
        "dataset":         "cambridgeltl/vsr_zeroshot",
        "split":           "validation",
        "n":               len(indices),
        "seed":            seed_used,
        "sequential":      sequential,
        "total_available": total,
        "indices":         indices,
        "created_at":      datetime.datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2)

    mode = "sequential first-N" if sequential else f"random (seed={seed})"
    print(f"Saved subset of {len(indices)}/{total} examples ({mode}) → {save_path}")
    return indices


def load_subset_indices(path: str) -> list[int]:
    """
    Load saved subset indices from a JSON file.

    Args:
        path: path to subset JSON (relative paths resolved against project root)
    Returns:
        list of HF dataset integer indices
    """
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    with open(path) as f:
        data = json.load(f)
    return data["indices"]


# ---------------------------------------------------------------------------
# CLI helper: create a subset from the command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create a matched VSR evaluation subset.")
    p.add_argument("--n",    type=int, default=100, help="Number of examples")
    p.add_argument("--seed", type=int, default=42,  help="RNG seed")
    p.add_argument("--out",        default=None,
                   help="Output path (default: data/subsets/vsr_n{n}_s{seed}.json or "
                        "data/subsets/vsr_n{n}_seq.json when --sequential)")
    p.add_argument("--sequential", action="store_true",
                   help="Use first-N indices instead of random sampling "
                        "(safer on HPC with partial HF cache)")
    args = p.parse_args()

    if args.out is None:
        if args.sequential:
            args.out = f"data/subsets/vsr_n{args.n}_seq.json"
        else:
            args.out = f"data/subsets/vsr_n{args.n}_s{args.seed}.json"

    create_subset(n=args.n, seed=args.seed, save_path=args.out, sequential=args.sequential)
