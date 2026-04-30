"""
Generate corrupted versions of VSR examples for robustness evaluation.

For each clean example, applies all 9 corruptions from the paper:
  - Gaussian blur   sigma in {1, 3, 5}                — RSA valid
  - Gaussian noise  variance in {0.001, 0.005, 0.010} — RSA valid
  - Rotation        degrees in {15, 45, 90}            — RSA invalid (geometry changes)

Output structure:
  <out_dir>/
    blur/sigma_1/      <example_id>.jpg
    blur/sigma_3/      <example_id>.jpg
    blur/sigma_5/      <example_id>.jpg
    noise/var_0.001/   <example_id>.jpg
    noise/var_0.005/   <example_id>.jpg
    noise/var_0.010/   <example_id>.jpg
    rotation/deg_15/   <example_id>.jpg
    rotation/deg_45/   <example_id>.jpg
    rotation/deg_90/   <example_id>.jpg
    metadata.json

Each entry in metadata.json:
  {
    "example_id":      "vsr_0000",
    "corruption_type": "blur",
    "severity":        1.0,
    "display_name":    "blur-1",
    "output_path":     "data/corrupted/blur/sigma_1/vsr_0000.jpg",
    "rsa_valid":       true
  }

RSA note:
  blur and noise do not change image geometry → reference boxes remain valid.
  rotation changes image coordinates → RSA must not be computed (rsa_valid=false).
  Box-coordinate transformation under rotation is not implemented; omitting RSA
  avoids reporting misleading numbers.

Usage:
  python src/generate_corrupted_dataset.py
  python src/generate_corrupted_dataset.py --n 20 --seed 42
  python src/generate_corrupted_dataset.py --n 20 --seed 42 --out-dir data/corrupted
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from corruptions import CorruptionSpec, apply_corruption, get_all_corruption_specs
from load_vsr import DEV_SUBSET_SIZE, load_vsr_dev_subset

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_OUT_DIR = os.path.join(_ROOT, "data", "corrupted")
DEFAULT_SEED    = 42


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------

def _noise_seed(base_seed: int, example_id: str) -> int:
    """
    Derive a deterministic per-example seed for Gaussian noise generation.

    Uses the numeric suffix of example_id ("vsr_0007" → 7) so each example
    always gets the same noise regardless of run order or batch size.
    """
    try:
        idx = int(example_id.rsplit("_", 1)[-1])
    except ValueError:
        idx = hash(example_id) & 0xFFFF
    return base_seed * 100_000 + idx


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_corrupted_dataset(
    n: int = DEV_SUBSET_SIZE,
    out_dir: str = DEFAULT_OUT_DIR,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """
    Load `n` clean VSR examples and write all corrupted versions to disk.

    Args:
        n:       number of VSR examples to process
        out_dir: root output directory (created if missing)
        seed:    base RNG seed for Gaussian noise reproducibility
    Returns:
        list of metadata dicts (one per generated image)
    """
    print(f"Loading {n} VSR examples...")
    examples = load_vsr_dev_subset(n=n)
    print(f"  {len(examples)} examples loaded.\n")

    specs = get_all_corruption_specs()
    metadata: list[dict] = []

    for i, ex in enumerate(examples):
        print(f"[{i + 1}/{len(examples)}] {ex.example_id}")
        image = ex.image.convert("RGB")  # ensure RGB before any corruption

        for spec in specs:
            # Derive noise seed per (example, spec) so results are reproducible
            # independently of iteration order.
            ex_seed = _noise_seed(seed, ex.example_id) if spec.corruption_type == "noise" else None

            corrupted = apply_corruption(image, spec, seed=ex_seed)

            # Build output path
            spec_dir = os.path.join(out_dir, spec.subdir)
            os.makedirs(spec_dir, exist_ok=True)
            out_path = os.path.join(spec_dir, f"{ex.example_id}.jpg")

            corrupted.save(out_path, format="JPEG", quality=95)

            # Compute relative path from project root for portability
            try:
                rel_path = os.path.relpath(out_path, _ROOT)
            except ValueError:
                rel_path = out_path  # fallback on Windows with different drives

            metadata.append({
                "example_id":      ex.example_id,
                "corruption_type": spec.corruption_type,
                "severity":        spec.severity,
                "display_name":    spec.display_name,
                "output_path":     rel_path,
                "rsa_valid":       spec.rsa_valid,
            })

        print(f"  {len(specs)} corrupted images saved → {os.path.join(out_dir, '<type>/.../')}")

    # Save metadata index
    metadata_path = os.path.join(out_dir, "metadata.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    n_rsa_valid   = sum(1 for m in metadata if m["rsa_valid"])
    n_rsa_invalid = len(metadata) - n_rsa_valid

    print(f"\nDone.")
    print(f"  Total images generated : {len(metadata)}")
    print(f"  RSA-valid entries      : {n_rsa_valid}  (blur + noise)")
    print(f"  RSA-invalid entries    : {n_rsa_invalid}  (rotation — geometry changes)")
    print(f"  Metadata saved         → {metadata_path}")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate corrupted VSR images for robustness evaluation."
    )
    parser.add_argument(
        "--n", type=int, default=DEV_SUBSET_SIZE,
        help=f"Number of VSR examples to process (default: {DEV_SUBSET_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base RNG seed for Gaussian noise (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_corrupted_dataset(n=args.n, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
