"""
Image corruption functions for robustness evaluation.

Implements the three corruption families from the paper:
  - Gaussian blur   : sigma in {1, 3, 5}                — geometry preserved → RSA valid
  - Gaussian noise  : variance in {0.001, 0.005, 0.010} — geometry preserved → RSA valid
  - Rotation        : degrees in {15, 45, 90}            — geometry changes   → RSA invalid

RSA note
--------
Blur and noise apply pixel-level perturbations that do not alter image dimensions
or the positions of objects; the original reference bounding boxes remain valid
and RSA (IoU >= 0.5) can still be computed.

Rotation changes the coordinate frame of every pixel.  Transforming bounding boxes
under an arbitrary rotation requires trigonometric re-projection that is not
implemented here.  To avoid reporting misleading RSA values, rsa_valid=False is set
for all rotation specs, and the evaluation pipeline should skip RSA for those cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Corruption spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorruptionSpec:
    """Describes one (type, severity) combination."""
    corruption_type: str  # "blur" | "noise" | "rotation"
    severity: float       # sigma, variance, or degrees as used in the paper
    display_name: str     # compact label, e.g. "blur-1", "noise-005", "rot-45"
    rsa_valid: bool       # whether original reference boxes are still valid
    subdir: str           # relative path under data/corrupted/


def is_rsa_valid_for_corruption(corruption_type: str) -> bool:
    """
    Return True if RSA can be meaningfully computed for this corruption type.

    Only blur and noise preserve image geometry.  Rotation does not, and since
    box-coordinate transformation under rotation is not implemented, RSA must
    not be reported for rotated images.
    """
    return corruption_type in ("blur", "noise")


def get_all_corruption_specs() -> list[CorruptionSpec]:
    """
    Return the 9 corruption specs matching the paper experimental setup:
      3 blur levels × 1 + 3 noise levels × 1 + 3 rotation levels × 1
    """
    specs: list[CorruptionSpec] = []

    # Gaussian blur: sigma in {1, 3, 5}
    for sigma in [1, 3, 5]:
        specs.append(CorruptionSpec(
            corruption_type="blur",
            severity=float(sigma),
            display_name=f"blur-{sigma}",
            rsa_valid=True,
            subdir=f"blur/sigma_{sigma}",
        ))

    # Gaussian noise: variance in {0.001, 0.005, 0.010}
    for var, tag in [(0.001, "001"), (0.005, "005"), (0.010, "010")]:
        specs.append(CorruptionSpec(
            corruption_type="noise",
            severity=var,
            display_name=f"noise-{tag}",
            rsa_valid=True,
            subdir=f"noise/var_{var:.3f}",
        ))

    # Rotation: degrees in {15, 45, 90}  —  RSA invalid (geometry changes)
    for deg in [15, 45, 90]:
        specs.append(CorruptionSpec(
            corruption_type="rotation",
            severity=float(deg),
            display_name=f"rot-{deg}",
            rsa_valid=False,
            subdir=f"rotation/deg_{deg}",
        ))

    return specs


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """
    Apply Gaussian blur.

    Args:
        image: input PIL image
        sigma: Gaussian standard deviation; paper values {1, 3, 5}
    Returns:
        blurred PIL image (same mode and size)
    """
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_gaussian_noise(
    image: Image.Image,
    variance: float,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Add per-pixel iid Gaussian noise.

    Processing steps:
      1. Convert to float32 in [0, 1]
      2. Sample noise ~ N(0, sqrt(variance)) per channel per pixel
      3. Add and clip to [0, 1]
      4. Convert back to uint8 RGB

    Args:
        image:    input PIL image
        variance: noise variance; paper values {0.001, 0.005, 0.010}
        seed:     RNG seed for reproducibility (None = unseeded)
    Returns:
        noisy RGB uint8 PIL image
    """
    rng = np.random.default_rng(seed)
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    noise = rng.normal(0.0, np.sqrt(variance), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def apply_rotation(
    image: Image.Image,
    degrees: float,
    expand: bool = False,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Rotate image counter-clockwise by the given angle.

    IMPORTANT: Rotation invalidates bounding box coordinates.
    Do NOT compute RSA for images produced by this function.
    The rsa_valid=False flag on CorruptionSpec enforces this.

    Args:
        image:      input PIL image
        degrees:    counter-clockwise angle; paper values {15, 45, 90}
        expand:     False (default) → output keeps original canvas size,
                    corners filled with fill_color.
                    True → canvas expands to contain the full rotated image.
        fill_color: RGB tuple for background fill (default: black)
    Returns:
        rotated PIL image
    """
    return image.rotate(degrees, expand=expand, fillcolor=fill_color)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def apply_corruption(
    image: Image.Image,
    spec: CorruptionSpec,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Apply the corruption described by `spec`.

    The `seed` argument is only used for noise corruptions; it is ignored
    for blur and rotation (which are deterministic).
    """
    if spec.corruption_type == "blur":
        return apply_gaussian_blur(image, spec.severity)
    elif spec.corruption_type == "noise":
        return apply_gaussian_noise(image, spec.severity, seed=seed)
    elif spec.corruption_type == "rotation":
        return apply_rotation(image, spec.severity)
    else:
        raise ValueError(f"Unknown corruption type: {spec.corruption_type!r}")
