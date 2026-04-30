"""
Visual CoT + Verification — two-pass inference pipeline (Setting 3).

Pass 1:  model sees image + caption → outputs reasoning, box, answer
Pass 2:  model sees image + caption + its own pass-1 box → verifies/revises
         box and outputs final answer

Both passes run sequentially per example in a single job.
The model is loaded once and shared across all examples and both passes.

Usage:
    cd /path/to/project
    python src/run_visual_cot_verification.py

Output:
    results/visual_cot_verification_results.json
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_vsr import load_vsr_dev_subset, DEV_SUBSET_SIZE
from model_interface import QwenVLLocalInterface
from metrics import compute_rsa, iou as compute_iou
from parse_outputs import parse_full_output, parse_reasoning, parse_answer, parse_box, extract_json_object
from utils import (
    load_prompt_template,
    format_prompt,
    format_prompt_multi,
    save_results,
    print_summary_vcot_verification,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EXAMPLES    = DEV_SUBSET_SIZE
RSA_THRESHOLD = 0.5
BOX_EPS       = 1e-4   # tolerance for comparing two boxes as "same"

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_PATH = os.path.join(_ROOT, "results", "visual_cot_verification_results.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boxes_differ(box_a, box_b) -> bool:
    """Return True if two boxes are meaningfully different (or one is None)."""
    if box_a is None and box_b is None:
        return False
    if box_a is None or box_b is None:
        return True
    return any(abs(a - b) > BOX_EPS for a, b in zip(box_a, box_b))


def _fmt_box(box) -> str:
    """Format a box for insertion into the pass-2 prompt."""
    if box is None:
        return "none (no valid box was identified in pass 1)"
    return str([round(v, 4) for v in box])


def _parse_pass1(raw: str) -> dict:
    """Parse pass-1 output (expects reasoning + box + answer)."""
    parsed_json = extract_json_object(raw)
    box_result  = parse_box(parsed_json or raw, mode="visual_cot")
    return {
        "pass1_raw_output"       : raw,
        "pass1_parsed_json"      : parsed_json,
        "pass1_reasoning"        : parse_reasoning(parsed_json or raw, key="reasoning"),
        "pass1_box"              : box_result["box"],
        "pass1_box_valid"        : box_result["valid"],
        "pass1_box_invalid_reason": box_result["invalid_reason"],
        "pass1_answer"           : parse_answer(parsed_json or raw),
    }


def _parse_pass2(raw: str) -> dict:
    """Parse pass-2 output (expects verification_reasoning + box + answer)."""
    parsed_json = extract_json_object(raw)
    box_result  = parse_box(parsed_json or raw, mode="visual_cot")
    return {
        "pass2_raw_output"              : raw,
        "pass2_parsed_json"             : parsed_json,
        "pass2_verification_reasoning"  : parse_reasoning(parsed_json or raw, key="verification_reasoning"),
        "pass2_box"                     : box_result["box"],
        "pass2_box_valid"               : box_result["valid"],
        "pass2_box_invalid_reason"      : box_result["invalid_reason"],
        "pass2_answer"                  : parse_answer(parsed_json or raw),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_visual_cot_verification() -> list[dict]:
    # ---- Load data ----
    print("Loading VSR dev subset...")
    examples = load_vsr_dev_subset(n=N_EXAMPLES)
    print(f"  {len(examples)} examples ready.\n")

    # ---- Load prompt templates ----
    pass1_template = load_prompt_template("visual_cot_pass1")
    pass2_template = load_prompt_template("visual_cot_verification_pass2")

    # ---- Load model (once, shared for both passes) ----
    model = QwenVLLocalInterface()

    results: list[dict] = []

    for i, ex in enumerate(examples):
        print(f"\n[{i + 1}/{len(examples)}] {ex.example_id}")
        print(f"  Caption : {ex.caption[:80]}{'...' if len(ex.caption) > 80 else ''}")
        gt_str = "true" if ex.label else "false"
        target_box = ex.target_box_normalized()

        # ----------------------------------------------------------------
        # PASS 1 — initial visual grounding + answer
        # ----------------------------------------------------------------
        p1_prompt = format_prompt(pass1_template, ex.caption)

        try:
            p1_raw = model.generate_response(ex.image, p1_prompt)
        except Exception as exc:
            print(f"  [Pass 1] ERROR: {exc}")
            p1_raw = ""

        p1 = _parse_pass1(p1_raw)
        p1_answer_correct = (p1["pass1_answer"] == gt_str) if p1["pass1_answer"] else False

        # Pass-1 spatial metrics
        p1_iou = p1_rsa = None
        if p1["pass1_box_valid"] and p1["pass1_box"] and target_box:
            p1_iou = compute_iou(p1["pass1_box"], target_box)
            p1_rsa = compute_rsa(p1["pass1_box"], target_box, RSA_THRESHOLD)

        print(f"  [P1] ans={p1['pass1_answer']}  correct={p1_answer_correct}"
              f"  box_valid={p1['pass1_box_valid']}"
              + (f"  iou={p1_iou:.3f}" if p1_iou is not None else ""))

        # ----------------------------------------------------------------
        # PASS 2 — verification / box revision
        # ----------------------------------------------------------------
        p2_prompt = format_prompt_multi(
            pass2_template,
            caption=ex.caption,
            initial_box=_fmt_box(p1["pass1_box"]),
        )

        try:
            p2_raw = model.generate_response(ex.image, p2_prompt)
        except Exception as exc:
            print(f"  [Pass 2] ERROR: {exc}")
            p2_raw = ""

        p2 = _parse_pass2(p2_raw)
        p2_answer_correct = (p2["pass2_answer"] == gt_str) if p2["pass2_answer"] else False

        # Pass-2 spatial metrics
        p2_iou = p2_rsa = None
        if p2["pass2_box_valid"] and p2["pass2_box"] and target_box:
            p2_iou = compute_iou(p2["pass2_box"], target_box)
            p2_rsa = compute_rsa(p2["pass2_box"], target_box, RSA_THRESHOLD)

        # Verification-specific flags
        box_revised    = _boxes_differ(p1["pass1_box"], p2["pass2_box"])
        answer_flipped = (
            p1["pass1_answer"] is not None
            and p2["pass2_answer"] is not None
            and p1["pass1_answer"] != p2["pass2_answer"]
        )
        recovered = (not p1_answer_correct) and p2_answer_correct

        print(f"  [P2] ans={p2['pass2_answer']}  correct={p2_answer_correct}"
              f"  box_revised={box_revised}  answer_flipped={answer_flipped}"
              + (f"  iou={p2_iou:.3f}" if p2_iou is not None else ""))

        # ----------------------------------------------------------------
        # Assemble result record
        # ----------------------------------------------------------------
        result: dict = {
            "example_id"              : ex.example_id,
            "caption"                 : ex.caption,
            "relation"                : ex.relation,
            "subj"                    : ex.subj,
            "obj"                     : ex.obj,
            "ground_truth_answer"     : gt_str,
            "target_box"              : target_box,
            # pass 1
            **p1,
            "pass1_answer_correct"    : p1_answer_correct,
            "pass1_iou"               : p1_iou,
            "pass1_rsa"               : p1_rsa,
            # pass 2
            **p2,
            "pass2_answer_correct"    : p2_answer_correct,
            "pass2_iou"               : p2_iou,
            "pass2_rsa"               : p2_rsa,
            # comparison flags
            "box_revised"             : box_revised,
            "answer_flipped"          : answer_flipped,
            "recovered"               : recovered,
        }
        results.append(result)

    # ---- Save ----
    save_results(results, RESULTS_PATH)

    # ---- Summary ----
    print_summary_vcot_verification(results)

    return results


if __name__ == "__main__":
    run_visual_cot_verification()
