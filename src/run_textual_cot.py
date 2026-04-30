"""
Textual CoT inference on the VSR dev subset.

What this script does:
  1. Load DEV_SUBSET_SIZE VSR examples from HuggingFace
  2. Load the textual_cot prompt template
  3. Load Qwen2.5-VL-7B-Instruct (local HF inference)
  4. For each example: run inference → parse JSON → compare to ground truth
  5. Save structured results to results/textual_cot_results.json
  6. Print a summary (FAA)

Usage:
    cd /path/to/project
    python src/run_textual_cot.py

Output:
    results/textual_cot_results.json
"""

from __future__ import annotations

import os
import sys

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_vsr import load_vsr_dev_subset, DEV_SUBSET_SIZE
from model_interface import QwenVLLocalInterface
from parse_outputs import parse_full_output
from utils import (
    load_prompt_template,
    format_prompt,
    save_results,
    print_summary_textual,
    log_example_result,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE = "textual_cot"
N_EXAMPLES = DEV_SUBSET_SIZE
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "textual_cot_results.json"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_textual_cot() -> list[dict]:
    # ---- Load data ----
    print("Loading VSR dev subset...")
    examples = load_vsr_dev_subset(n=N_EXAMPLES)
    print(f"  {len(examples)} examples ready.\n")

    # ---- Load prompt ----
    prompt_template = load_prompt_template("textual_cot")

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

        # Target box (for completeness; not evaluated in textual_cot)
        target_box = ex.target_box_normalized()

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
            # --- box fields (should be null for textual_cot) ---
            "parsed_box": parsed["parsed_box"],
            "box_valid": parsed["box_valid"],
            "box_invalid_reason": parsed["box_invalid_reason"],
            # --- spatial eval (n/a for textual_cot) ---
            "target_box": target_box,
            "iou": None,
            "rsa": None,
        }
        results.append(result)

        log_example_result(result, mode=MODE)

    # ---- Persist ----
    save_results(results, RESULTS_PATH)

    # ---- Summary ----
    print_summary_textual(results)

    return results


if __name__ == "__main__":
    run_textual_cot()
