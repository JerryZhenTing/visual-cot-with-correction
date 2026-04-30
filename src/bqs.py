"""
Box Quality Score (BQS) for the multi-stage correction pipeline.

BQS is a research-time heuristic that estimates how reliable an initial
predicted box is.  It is used to decide whether correction should be triggered.

Formula (when box is valid):
    BQS = 0.15 * format_score
        + 0.50 * overlap_score
        + 0.15 * mention_score
        + 0.20 * stability_score

Components:
    format_score   : 1.0 if box passes structural validation, else 0.0
    overlap_score  : IoU between predicted box and ground-truth reference region
    mention_score  : 1.0 if both objects named in reasoning, 0.5 if one, 0.0 if neither
    stability_score: 1.0 if final answer matches initial answer (post-hoc),
                     placeholder 1.0 before the crop pass runs

If box is invalid:  BQS = 0 unconditionally (hard fail).

Note: overlap_score uses ground-truth reference boxes (union of COCO bboxes for
the two relevant objects).  This is allowed because BQS is a development-time
diagnostic, not a model input.

Trigger rule (applied in run_multistage_correction.py):
    trigger correction  if box is invalid
    trigger correction  if BQS < BQS_THRESHOLD (default 0.6)
    keep original       otherwise
"""

from __future__ import annotations

from typing import Optional

from metrics import iou

# Component weights — must sum to 1.0
_W_FORMAT    = 0.15
_W_OVERLAP   = 0.50
_W_MENTION   = 0.15
_W_STABILITY = 0.20


def _mention_score(
    reasoning: Optional[str],
    subj: Optional[str],
    obj: Optional[str],
) -> float:
    """
    Score how well the reasoning mentions the relevant objects.

    Returns:
        1.0 if both subj and obj appear in reasoning (case-insensitive)
        0.5 if exactly one appears
        0.0 if neither appears or reasoning is None
    """
    if not reasoning:
        return 0.0
    text = reasoning.lower()
    subj_hit = bool(subj) and subj.lower() in text
    obj_hit  = bool(obj)  and obj.lower()  in text
    if subj_hit and obj_hit:
        return 1.0
    if subj_hit or obj_hit:
        return 0.5
    return 0.0


def compute_bqs(
    box_valid: bool,
    box: Optional[list],
    target_box: Optional[list],
    reasoning: Optional[str],
    subj: Optional[str],
    obj: Optional[str],
    stability_score: float = 1.0,
) -> dict:
    """
    Compute the Box Quality Score for one predicted box.

    Args:
        box_valid:       whether the predicted box passed structural validation
        box:             predicted [xmin, ymin, xmax, ymax] in [0,1], or None
        target_box:      reference region (union of COCO object boxes), or None
        reasoning:       model reasoning string from pass 1
        subj:            subject object name from VSR annotation (e.g. "cat")
        obj:             object name from VSR annotation (e.g. "dog")
        stability_score: 1.0 if final answer == initial answer, else 0.0;
                         use 1.0 as a placeholder before the crop pass runs

    Returns:
        dict with keys: bqs, format_score, overlap_score, mention_score,
                        stability_score
    """
    if not box_valid:
        # Invalid box format → hard fail
        return {
            "bqs":             0.0,
            "format_score":    0.0,
            "overlap_score":   0.0,
            "mention_score":   _mention_score(reasoning, subj, obj),
            "stability_score": stability_score,
        }

    format_score = 1.0

    # Overlap: IoU against reference region (None when target_box unavailable)
    if box is not None and target_box is not None:
        overlap_score = iou(box, target_box)
    else:
        overlap_score = 0.0

    mention = _mention_score(reasoning, subj, obj)

    score = (
        _W_FORMAT    * format_score
        + _W_OVERLAP   * overlap_score
        + _W_MENTION   * mention
        + _W_STABILITY * stability_score
    )

    return {
        "bqs":             round(score, 6),
        "format_score":    format_score,
        "overlap_score":   round(overlap_score, 6),
        "mention_score":   mention,
        "stability_score": stability_score,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Perfect box, perfect overlap, both objects mentioned, stable
    res = compute_bqs(
        box_valid=True,
        box=[0.0, 0.0, 1.0, 1.0],
        target_box=[0.0, 0.0, 1.0, 1.0],
        reasoning="The cat is to the left of the dog.",
        subj="cat",
        obj="dog",
        stability_score=1.0,
    )
    assert abs(res["bqs"] - 1.0) < 1e-6, res
    assert res["mention_score"] == 1.0
    assert res["overlap_score"] == 1.0

    # Invalid box → BQS = 0
    res2 = compute_bqs(box_valid=False, box=None, target_box=None,
                       reasoning=None, subj="cat", obj="dog")
    assert res2["bqs"] == 0.0
    assert res2["format_score"] == 0.0

    # One object mentioned
    res3 = compute_bqs(
        box_valid=True,
        box=[0.1, 0.1, 0.5, 0.5],
        target_box=[0.1, 0.1, 0.5, 0.5],
        reasoning="I can see the cat here.",
        subj="cat",
        obj="dog",
        stability_score=1.0,
    )
    assert res3["mention_score"] == 0.5

    # No target box → overlap = 0
    res4 = compute_bqs(
        box_valid=True,
        box=[0.1, 0.1, 0.5, 0.5],
        target_box=None,
        reasoning="The cat is above the dog.",
        subj="cat",
        obj="dog",
        stability_score=1.0,
    )
    assert res4["overlap_score"] == 0.0

    print("bqs smoke tests passed.")
