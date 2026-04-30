"""
Visual CoT inference on the VSR dev subset.

What this script does:
  1. Load DEV_SUBSET_SIZE VSR examples from HuggingFace
  2. Load the visual_cot prompt template
  3. Load Qwen2.5-VL-7B-Instruct (local HF inference)
  4. For each example: run inference → parse JSON → compare to ground truth
                       → validate predicted box → compute IoU / RSA vs target box
  5. Save structured results to results/visual_cot_results.json
  6. Print a summary (FAA, valid box rate, mean RSA)

Usage:
    cd /path/to/project
    python src/run_visual_cot.py

Output:
    results/visual_cot_results.json
"""

from __future__ import annotations

import os
import sys

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_vsr import load_vsr_dev_subset, DEV_SUBSET_SIZE
from model_interface import QwenVLLocalInterface
from metrics import compute_rsa, iou as compute_iou
from parse_outputs import parse_full_output
from utils import (
    load_prompt_template,
    format_prompt,
    save_results,
    print_summary_visual,
    log_example_result,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE = "visual_cot"
N_EXAMPLES = DEV_SUBSET_SIZE
RSA_THRESHOLD = 0.5
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "visual_cot_results.json"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_visual_cot() -> list[dict]:
    # ---- Load data ----
    print("Loading VSR dev subset...")
    examples = load_vsr_dev_subset(n=N_EXAMPLES)
    print(f"  {len(examples)} examples ready.\n")

    # ---- Load prompt ----
    prompt_template = load_prompt_template("visual_cot")

    # ---- Load model ----
    model = QwenVLLocalInterface()

    results: list[dict] = []

    for i, ex in enumerate(examples):
        print(f"\n[{i + 1}/{len(examples)}] {ex.example_id}")
        print(f"  Caption : {ex.caption[:80]}{'...' if len(ex.caption) > 80 else ''}")
        print(f"  GT      : {ex.label}")

        # Format prompt for this example
        prompt = format_prompt(prompt_template, ex.caption)

        # Inference
        try:
            raw_output = model.generate_response(ex.image, prompt)
        except Exception as exc:
            print(f"  ERROR during inference: {exc}")
            raw_output = ""

        # Show raw output (truncated)
        preview = raw_output.replace("\n", " ")[:200]
        print(f"  Raw     : {preview}")

        # Parse JSON output
        parsed = parse_full_output(raw_output, mode=MODE)

        # Evaluate answer against ground truth
        gt_str = "true" if ex.label else "false"
        parsed_ans = parsed["parsed_answer"]
        answer_correct = (parsed_ans == gt_str) if parsed_ans is not None else False

        # Target box: normalized union of obj1+obj2 (may be None)
        target_box = ex.target_box_normalized()
        pred_box = parsed["parsed_box"]

        # IoU and RSA (computed only when both boxes are valid)
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
            "raw_output": raw_output,
            # --- parsed fields ---
            "parsed_json": parsed["parsed_json"],
            "parsed_reasoning": parsed["parsed_reasoning"],
            "parsed_answer": parsed_ans,
            "answer_correct": answer_correct,
            # --- box fields ---
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
    print_summary_visual(results)

    return results


if __name__ == "__main__":
    run_visual_cot()
