"""
Visual CoT + Verification inference on the VSR dev subset (Setting 3).

What this script does:
  1. Load VSR examples (for images)
  2. Load existing visual_cot_results.json (initial pass — no re-inference)
  3. For each example:
       a. Build a verification prompt using the model's own previous
          reasoning, box, and answer as context
       b. Run a second inference pass on the same image
       c. Parse the verification response
       d. Compare verified answer to ground truth
       e. Track whether verification changed the answer
  4. Save to results/verification_results.json
  5. Print summary: initial FAA vs verified FAA, answer change rate,
     valid box rate, mean RSA

Usage:
    cd /path/to/project
    python src/run_verification.py

Prerequisites:
    results/visual_cot_results.json must exist (run run_visual_cot.py first)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_vsr import load_vsr_dev_subset, DEV_SUBSET_SIZE
from model_interface import QwenVLLocalInterface
from metrics import compute_rsa, iou as compute_iou
from parse_outputs import parse_full_output, parse_field
from utils import (
    load_prompt_template,
    format_prompt_multi,
    save_results,
    print_summary_verification,
    log_example_result,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE = "verification"
N_EXAMPLES = DEV_SUBSET_SIZE
RSA_THRESHOLD = 0.5

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SRC_DIR, "..")

VISUAL_COT_RESULTS_PATH = os.path.join(_PROJECT_ROOT, "results", "visual_cot_results.json")
RESULTS_PATH = os.path.join(_PROJECT_ROOT, "results", "verification_results.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_visual_cot_results(path: str) -> dict[str, dict]:
    """Load visual CoT results and index by example_id."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Visual CoT results not found at {path}\n"
            "Run run_visual_cot.py first."
        )
    with open(path) as f:
        results = json.load(f)
    return {r["example_id"]: r for r in results}


def _format_box_for_prompt(box) -> str:
    """Format a box (or None) as a readable string for the prompt."""
    if box is None:
        return "none (no valid box was identified)"
    return str([round(v, 3) for v in box])


def _format_reasoning_for_prompt(reasoning) -> str:
    if not reasoning:
        return "none provided"
    return reasoning


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_verification() -> list[dict]:
    # ---- Load initial Visual CoT results ----
    print("Loading Visual CoT results...")
    initial_results = _load_visual_cot_results(VISUAL_COT_RESULTS_PATH)
    print(f"  {len(initial_results)} initial results loaded.")

    # ---- Load VSR examples (need images for second inference pass) ----
    print("Loading VSR dev subset (for images)...")
    examples = load_vsr_dev_subset(n=N_EXAMPLES)
    print(f"  {len(examples)} examples ready.\n")

    # ---- Load prompt template ----
    prompt_template = load_prompt_template("verification")

    # ---- Load model ----
    model = QwenVLLocalInterface()

    results: list[dict] = []

    for i, ex in enumerate(examples):
        print(f"\n[{i + 1}/{len(examples)}] {ex.example_id}")
        print(f"  Caption : {ex.caption[:80]}{'...' if len(ex.caption) > 80 else ''}")

        # Get the initial Visual CoT output for this example
        init = initial_results.get(ex.example_id, {})
        init_answer = init.get("parsed_answer")
        init_box = init.get("parsed_box")
        init_reasoning = init.get("parsed_reasoning")
        init_answer_correct = init.get("answer_correct", False)

        print(f"  Initial : ans={init_answer}  box={_format_box_for_prompt(init_box)}")

        # Build verification prompt with previous context
        prompt = format_prompt_multi(
            prompt_template,
            caption=ex.caption,
            prev_reasoning=_format_reasoning_for_prompt(init_reasoning),
            prev_box=_format_box_for_prompt(init_box),
            prev_answer=init_answer if init_answer else "none",
        )

        # Run second inference pass
        try:
            raw_output = model.generate_response(ex.image, prompt)
        except Exception as exc:
            print(f"  ERROR during inference: {exc}")
            raw_output = ""

        preview = raw_output.replace("\n", " ")[:200]
        print(f"  Raw     : {preview}")

        # Parse verification output (same structure as visual_cot: box required)
        parsed = parse_full_output(raw_output, mode="visual_cot")

        # Extract the "verification" text field (replaces "reasoning" in this prompt)
        verification_text = parse_field(parsed["parsed_json"], "verification")

        # Ground truth
        gt_str = "true" if ex.label else "false"
        verified_answer = parsed["parsed_answer"]
        answer_correct = (verified_answer == gt_str) if verified_answer is not None else False

        # Did verification change the answer?
        answer_changed = (
            init_answer is not None
            and verified_answer is not None
            and init_answer != verified_answer
        )

        # Target box and spatial metrics
        target_box = ex.target_box_normalized()
        pred_box = parsed["parsed_box"]

        iou_score: float | None = None
        rsa_score: float | None = None
        if parsed["box_valid"] and pred_box is not None and target_box is not None:
            iou_score = compute_iou(pred_box, target_box)
            rsa_score = compute_rsa(pred_box, target_box, threshold=RSA_THRESHOLD)

        result: dict = {
            "example_id": ex.example_id,
            "caption": ex.caption,
            "relation": ex.relation,
            "subj": ex.subj,
            "obj": ex.obj,
            "ground_truth": gt_str,
            # --- initial Visual CoT pass (for comparison) ---
            "initial_answer": init_answer,
            "initial_answer_correct": init_answer_correct,
            "initial_box": init_box,
            "initial_reasoning": init_reasoning,
            # --- verification pass ---
            "raw_output": raw_output,
            "parsed_json": parsed["parsed_json"],
            "verification_text": verification_text,
            "parsed_reasoning": parsed["parsed_reasoning"],  # fallback if model uses "reasoning" key
            "parsed_answer": verified_answer,
            "answer_correct": answer_correct,
            "answer_changed": answer_changed,
            # --- box ---
            "parsed_box": pred_box,
            "box_valid": parsed["box_valid"],
            "box_invalid_reason": parsed["box_invalid_reason"],
            # --- spatial eval ---
            "target_box": target_box,
            "iou": iou_score,
            "rsa": rsa_score,
        }
        results.append(result)

        log_example_result(result, mode=MODE)

    # ---- Persist ----
    save_results(results, RESULTS_PATH)

    # ---- Summary ----
    print_summary_verification(results)

    return results


if __name__ == "__main__":
    run_verification()
