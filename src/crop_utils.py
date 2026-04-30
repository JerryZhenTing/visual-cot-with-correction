"""
Image cropping utilities for the multi-stage correction pipeline.

All functions accept normalized [xmin, ymin, xmax, ymax] boxes in [0, 1].
"""

from __future__ import annotations

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
