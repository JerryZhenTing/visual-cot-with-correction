"""
Visual CoT + Multi-Stage Correction — Setting 4.

Pipeline per example:
  Stage 1 : Initial grounding   — model outputs reasoning, box, answer
  Stage 2 : Box Quality Score   — heuristic combining format, overlap, mention, stability
  Stage 3 : Correction trigger  — if box invalid OR BQS < threshold
  Stage 4 : Verification        — model revises box and gives updated answer  [if triggered]
  Stage 5 : Crop answering      — model answers from cropped region           [if triggered]

If correction is NOT triggered, the initial answer is the final answer.
If correction IS triggered, the crop-conditioned answer is the final answer.

BQS note: stability_score (answer unchanged after crop pass) is post-hoc.
  - Before the crop pass: stability = 1.0 (placeholder, optimistic)
  - After the crop pass:  stability = 1 if crop_answer == initial_answer else 0
  - BQS stored in results uses the final stability value.
  - Trigger decision uses the provisional BQS (stability = 1.0).

Output:
  results/multistage_correction_results.json

Usage:
  cd /path/to/project
  python src/run_multistage_correction.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bqs import compute_bqs
from crop_utils import safe_crop
from load_vsr import DEV_SUBSET_SIZE, load_vsr_dev_subset
from metrics import compute_rsa, iou as compute_iou
from model_interface import QwenVLLocalInterface
from parse_outputs import extract_json_object, parse_answer, parse_box, parse_reasoning
from utils import (
    format_prompt,
    format_prompt_multi,
    load_prompt_template,
    print_summary_multistage,
    save_results,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EXAMPLES    = DEV_SUBSET_SIZE
RSA_THRESHOLD = 0.5
BQS_THRESHOLD = 0.6      # correction triggered when provisional BQS < this
BOX_EPS       = 1e-4     # tolerance for "box revised" comparison

_ROOT        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_PATH = os.path.join(_ROOT, "results", "multistage_correction_results.json")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_stage1(raw: str) -> dict:
    """Parse pass-1 output: reasoning, box, answer."""
    pj = extract_json_object(raw)
    br = parse_box(pj or raw, mode="visual_cot")
    return {
        "initial_raw_output":         raw,
        "initial_parsed_json":        pj,
        "initial_reasoning":          parse_reasoning(pj or raw, key="reasoning"),
        "initial_box":                br["box"],
        "initial_box_valid":          br["valid"],
        "initial_box_invalid_reason": br["invalid_reason"],
        "initial_answer":             parse_answer(pj or raw),
    }


def _parse_verification(raw: str) -> dict:
    """Parse verification output: verification_reasoning, box, answer."""
    pj = extract_json_object(raw)
    br = parse_box(pj or raw, mode="visual_cot")
    return {
        "verification_raw_output":         raw,
        "verification_parsed_json":        pj,
        "verification_reasoning":          parse_reasoning(pj or raw, key="verification_reasoning"),
        "revised_box":                     br["box"],
        "revised_box_valid":               br["valid"],
        "revised_box_invalid_reason":      br["invalid_reason"],
        "revised_answer":                  parse_answer(pj or raw),
    }


def _parse_crop(raw: str) -> dict:
    """Parse crop-conditioned output: crop_reasoning, answer."""
    pj = extract_json_object(raw)
    return {
        "crop_raw_output":    raw,
        "crop_parsed_json":   pj,
        "crop_reasoning":     parse_reasoning(pj or raw, key="crop_reasoning"),
        "crop_answer":        parse_answer(pj or raw),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _boxes_differ(a, b) -> bool:
    """True if two boxes are meaningfully different (or one is None and the other isn't)."""
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return any(abs(x - y) > BOX_EPS for x, y in zip(a, b))


def _fmt_box(box) -> str:
    """Format a box for prompt insertion."""
    if box is None:
        return "none (no valid box was produced)"
    return str([round(v, 4) for v in box])


def _spatial_metrics(box_valid, box, target_box) -> tuple:
    """Return (iou, rsa) or (None, None) if either box is unavailable."""
    if box_valid and box and target_box:
        iou_val = compute_iou(box, target_box)
        rsa_val = compute_rsa(box, target_box, RSA_THRESHOLD)
        return iou_val, rsa_val
    return None, None


# ---------------------------------------------------------------------------
# Per-example pipeline
# ---------------------------------------------------------------------------

def _process_example(ex, model, p1_tmpl, verif_tmpl, crop_tmpl) -> dict:
    gt         = "true" if ex.label else "false"
    target_box = ex.target_box_normalized()

    # ------------------------------------------------------------------
    # Stage 1: Initial grounding
    # ------------------------------------------------------------------
    p1_prompt = format_prompt(p1_tmpl, ex.caption)
    try:
        p1_raw = model.generate_response(ex.image, p1_prompt)
    except Exception as exc:
        print(f"  [S1] ERROR: {exc}")
        p1_raw = ""

    s1 = _parse_stage1(p1_raw)
    initial_correct = (s1["initial_answer"] == gt) if s1["initial_answer"] else False
    initial_iou, initial_rsa = _spatial_metrics(
        s1["initial_box_valid"], s1["initial_box"], target_box
    )

    print(f"  [S1] ans={s1['initial_answer']}  correct={initial_correct}"
          f"  box_valid={s1['initial_box_valid']}"
          + (f"  iou={initial_iou:.3f}" if initial_iou is not None else ""))

    # ------------------------------------------------------------------
    # Stage 2: BQS (provisional — stability placeholder = 1.0)
    # ------------------------------------------------------------------
    bqs_provisional = compute_bqs(
        box_valid=s1["initial_box_valid"],
        box=s1["initial_box"],
        target_box=target_box,
        reasoning=s1["initial_reasoning"],
        subj=ex.subj,
        obj=ex.obj,
        stability_score=1.0,  # placeholder; recomputed after crop pass
    )

    # ------------------------------------------------------------------
    # Stage 3: Correction trigger
    # ------------------------------------------------------------------
    correction_triggered = (
        not s1["initial_box_valid"]
        or bqs_provisional["bqs"] < BQS_THRESHOLD
    )

    print(f"  [S2] bqs={bqs_provisional['bqs']:.3f}"
          f"  (fmt={bqs_provisional['format_score']:.2f}"
          f"  ovlp={bqs_provisional['overlap_score']:.3f}"
          f"  mntn={bqs_provisional['mention_score']:.2f})"
          f"  trigger={correction_triggered}")

    # ------------------------------------------------------------------
    # Stages 4 & 5: Correction path (only if triggered)
    # ------------------------------------------------------------------
    if correction_triggered:
        # Stage 4: Verification / box regeneration
        verif_prompt = format_prompt_multi(
            verif_tmpl,
            caption=ex.caption,
            initial_box=_fmt_box(s1["initial_box"]),
        )
        try:
            verif_raw = model.generate_response(ex.image, verif_prompt)
        except Exception as exc:
            print(f"  [S4] ERROR: {exc}")
            verif_raw = ""

        s4 = _parse_verification(verif_raw)
        revised_iou, revised_rsa = _spatial_metrics(
            s4["revised_box_valid"], s4["revised_box"], target_box
        )

        print(f"  [S4] ans={s4['revised_answer']}  box_valid={s4['revised_box_valid']}"
              + (f"  iou={revised_iou:.3f}" if revised_iou is not None else ""))

        # Stage 5: Crop-conditioned answering
        # Use revised box if valid; fall back to full image if not
        crop_img   = safe_crop(ex.image, s4["revised_box"] if s4["revised_box_valid"] else None)
        crop_prompt = format_prompt(crop_tmpl, ex.caption)
        try:
            crop_raw = model.generate_response(crop_img, crop_prompt)
        except Exception as exc:
            print(f"  [S5] ERROR: {exc}")
            crop_raw = ""

        s5          = _parse_crop(crop_raw)
        final_answer = s5["crop_answer"]

        # Recompute stability_score now that we have the crop answer
        stability_score = 1.0 if final_answer == s1["initial_answer"] else 0.0

        # Final BQS with real stability_score
        bqs_final = compute_bqs(
            box_valid=s1["initial_box_valid"],
            box=s1["initial_box"],
            target_box=target_box,
            reasoning=s1["initial_reasoning"],
            subj=ex.subj,
            obj=ex.obj,
            stability_score=stability_score,
        )

        print(f"  [S5] crop_ans={final_answer}  stab={stability_score:.1f}"
              f"  final_bqs={bqs_final['bqs']:.3f}")

    else:
        # Not triggered — skip verification and crop passes
        s4          = None
        s5          = None
        verif_raw   = None
        crop_raw    = None
        revised_iou = revised_rsa = None
        final_answer = s1["initial_answer"]
        stability_score = 1.0  # answer unchanged (no crop pass ran)
        bqs_final   = bqs_provisional  # stability was already 1.0

    final_answer_correct = (final_answer == gt) if final_answer else False

    # Derived flags
    revised_box = s4["revised_box"] if s4 else None
    box_revised    = _boxes_differ(s1["initial_box"], revised_box)
    answer_flipped = (
        s1["initial_answer"] is not None
        and final_answer is not None
        and s1["initial_answer"] != final_answer
    )
    recovered = (not initial_correct) and final_answer_correct

    print(f"  [FN] final_ans={final_answer}  correct={final_answer_correct}"
          f"  revised={box_revised}  flipped={answer_flipped}")

    return {
        "example_id":              ex.example_id,
        "caption":                 ex.caption,
        "relation":                ex.relation,
        "subj":                    ex.subj,
        "obj":                     ex.obj,
        "ground_truth_answer":     gt,
        "target_box":              target_box,
        # Stage 1
        "initial_raw_output":         s1["initial_raw_output"],
        "initial_parsed_json":        s1["initial_parsed_json"],
        "initial_reasoning":          s1["initial_reasoning"],
        "initial_box":                s1["initial_box"],
        "initial_box_valid":          s1["initial_box_valid"],
        "initial_box_invalid_reason": s1["initial_box_invalid_reason"],
        "initial_answer":             s1["initial_answer"],
        "initial_answer_correct":     initial_correct,
        "initial_iou":                initial_iou,
        "initial_rsa":                initial_rsa,
        # BQS
        "bqs":                     bqs_final["bqs"],
        "format_score":            bqs_final["format_score"],
        "overlap_score":           bqs_final["overlap_score"],
        "mention_score":           bqs_final["mention_score"],
        "stability_score":         bqs_final["stability_score"],
        "correction_triggered":    correction_triggered,
        # Stage 4: Verification
        "verification_raw_output":         verif_raw,
        "verification_parsed_json":        s4["verification_parsed_json"] if s4 else None,
        "verification_reasoning":          s4["verification_reasoning"] if s4 else None,
        "revised_box":                     s4["revised_box"] if s4 else None,
        "revised_box_valid":               s4["revised_box_valid"] if s4 else None,
        "revised_box_invalid_reason":      s4["revised_box_invalid_reason"] if s4 else None,
        "revised_answer":                  s4["revised_answer"] if s4 else None,
        "revised_iou":                     revised_iou,
        "revised_rsa":                     revised_rsa,
        # Stage 5: Crop
        "crop_raw_output":  crop_raw,
        "crop_parsed_json": s5["crop_parsed_json"] if s5 else None,
        "crop_reasoning":   s5["crop_reasoning"] if s5 else None,
        # Final
        "final_answer":         final_answer,
        "final_answer_correct": final_answer_correct,
        # Flags
        "box_revised":    box_revised,
        "answer_flipped": answer_flipped,
        "recovered":      recovered,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multistage_correction() -> list[dict]:
    print("Loading VSR dev subset...")
    examples = load_vsr_dev_subset(n=N_EXAMPLES)
    print(f"  {len(examples)} examples ready.\n")

    p1_tmpl    = load_prompt_template("multi_stage_pass1")
    verif_tmpl = load_prompt_template("multi_stage_verification")
    crop_tmpl  = load_prompt_template("multi_stage_crop_answer")

    model = QwenVLLocalInterface()

    results: list[dict] = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        print(f"  Caption : {ex.caption[:80]}{'...' if len(ex.caption) > 80 else ''}")
        result = _process_example(ex, model, p1_tmpl, verif_tmpl, crop_tmpl)
        results.append(result)

    save_results(results, RESULTS_PATH)
    print_summary_multistage(results)

    return results


if __name__ == "__main__":
    run_multistage_correction()
