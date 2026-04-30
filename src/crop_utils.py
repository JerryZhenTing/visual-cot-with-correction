"""
Image cropping utilities for the multi-stage correction pipeline.

All functions accept normalized [xmin, ymin, xmax, ymax] boxes in [0, 1].
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image


def clip_box(box: list) -> list:
    """Clip all coordinates to [0, 1]."""
    return [max(0.0, min(1.0, v)) for v in box]


def crop_normalized(
    image: Image.Image,
    box: list,
    min_size: int = 1,
) -> Image.Image:
    """
    Crop a PIL image using a normalized [xmin, ymin, xmax, ymax] box.

    Coordinates are clipped to [0, 1] before conversion to pixels.
    Returns the full image if the resulting crop would be degenerate
    (width or height < min_size pixels).

    Args:
        image:    input PIL image
        box:      [xmin, ymin, xmax, ymax] in [0, 1]
        min_size: minimum crop dimension in pixels (default 1)
    Returns:
        cropped PIL image, or original image if crop is degenerate
    """
    W, H = image.size
    xmin, ymin, xmax, ymax = clip_box(box)

    px_xmin = int(xmin * W)
    px_ymin = int(ymin * H)
    px_xmax = int(xmax * W)
    px_ymax = int(ymax * H)

    # Guarantee ordering after int truncation
    px_xmin, px_xmax = min(px_xmin, px_xmax), max(px_xmin, px_xmax)
    px_ymin, px_ymax = min(px_ymin, px_ymax), max(px_ymin, px_ymax)

    if (px_xmax - px_xmin) < min_size or (px_ymax - px_ymin) < min_size:
        # Degenerate box — fall back to full image
        return image

    return image.crop((px_xmin, px_ymin, px_xmax, px_ymax))


def safe_crop(
    image: Image.Image,
    box: Optional[list],
) -> Image.Image:
    """
    Crop using a normalized box, falling back to the full image if box is None
    or degenerate.

    This is the primary entry point used by the multi-stage pipeline.

    Args:
        image: input PIL image
        box:   [xmin, ymin, xmax, ymax] in [0, 1], or None
    Returns:
        cropped (or original) PIL image
    """
    if box is None:
        return image
    try:
        return crop_normalized(image, box)
    except Exception:
        return image


# ---------------------------------------------------------------------------
# Additional utilities for guidance pipeline
# ---------------------------------------------------------------------------

def normalized_box_to_pixel_box(
    box: list,
    image_width: int,
    image_height: int,
) -> list[int]:
    """Convert normalized [0,1] box to integer pixel [xmin,ymin,xmax,ymax]."""
    xmin, ymin, xmax, ymax = clip_box(box)
    return [
        int(xmin * image_width),
        int(ymin * image_height),
        int(xmax * image_width),
        int(ymax * image_height),
    ]


def pixel_box_to_normalized_box(
    box: list,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert integer pixel box to normalized [0,1] [xmin,ymin,xmax,ymax]."""
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / image_width)),
        max(0.0, min(1.0, y1 / image_height)),
        max(0.0, min(1.0, x2 / image_width)),
        max(0.0, min(1.0, y2 / image_height)),
    ]


def crop_image_from_normalized_box(
    image: Image.Image,
    box: Optional[list],
) -> Image.Image:
    """Crop image using a normalized box. Falls back to full image on invalid input."""
    return safe_crop(image, box)


def save_crop(
    image: Image.Image,
    box: Optional[list],
    output_path: str,
) -> None:
    """Crop and save to output_path. Creates parent dirs automatically."""
    crop = safe_crop(image, box)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    crop.save(output_path)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    # Normal crop
    cropped = crop_normalized(img, [0.1, 0.2, 0.6, 0.8])
    assert cropped.size == (50, 60), f"Expected (50, 60), got {cropped.size}"

    # Degenerate box → full image returned
    degenerate = crop_normalized(img, [0.5, 0.5, 0.5, 0.5])
    assert degenerate.size == (100, 100)

    # Out-of-range coords → clipped
    clipped = crop_normalized(img, [-0.1, -0.1, 1.5, 1.5])
    assert clipped.size == (100, 100)

    # None box → full image
    result = safe_crop(img, None)
    assert result.size == (100, 100)

    print("crop_utils smoke tests passed.")
