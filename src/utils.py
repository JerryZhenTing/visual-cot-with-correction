"""
General project utilities: prompt loading, formatting, result persistence,
and summary printing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def load_prompt_template(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts/ directory.

    Args:
        prompt_name: filename stem without .txt (e.g. "textual_cot")
    Returns:
        template string containing the {caption} placeholder
    Raises:
        FileNotFoundError if the file does not exist
    """
    # src/ → project root → prompts/
    project_root = Path(__file__).resolve().parent.parent
    prompt_path = project_root / "prompts" / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    return prompt_path.read_text().strip()


def format_prompt(template: str, caption: str) -> str:
    """
    Substitute {caption} in a prompt template.

    Args:
        template: prompt template string
        caption:  VSR caption / statement for this example
    Returns:
        fully-formatted prompt ready to send to the model
    """
    return template.replace("{caption}", caption)


def format_prompt_multi(template: str, **kwargs) -> str:
    """
    Substitute multiple {key} placeholders in a prompt template.

    Args:
        template: prompt template string with {key} placeholders
        **kwargs: key=value pairs to substitute
    Returns:
        fully-formatted prompt string
    Example:
        format_prompt_multi(t, caption="...", prev_answer="true", prev_box="[0.1,...]")
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_path: str) -> None:
    """
    Save a list of result dicts to a JSON file.
    Creates parent directories as needed.

    Args:
        results:     list of per-example result dicts
        output_path: path to the output .json file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"Saved {len(results)} results → {output_path}")


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for numpy arrays and similar objects."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def print_summary_textual(results: list[dict]) -> None:
    """Print a short summary table for a Textual CoT run."""
    from metrics import final_answer_accuracy  # local import avoids circular dep at load time

    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    faa = final_answer_accuracy(results)
    n_correct = sum(1 for r in results if r.get("answer_correct", False))
    n_parsed = sum(1 for r in results if r.get("parsed_answer") is not None)

    print("\n" + "=" * 52)
    print("  Textual CoT — Summary")
    print("=" * 52)
    print(f"  Examples          : {n}")
    print(f"  Parsed answers    : {n_parsed} / {n}")
    print(f"  Final Answer Acc  : {faa:.3f}   ({n_correct}/{n} correct)")
    print("=" * 52)


def print_summary_visual(results: list[dict]) -> None:
    """Print a short summary table for a Visual CoT run."""
    from metrics import final_answer_accuracy, valid_box_rate, mean_rsa

    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    faa = final_answer_accuracy(results)
    vbr = valid_box_rate(results)
    mrsa = mean_rsa(results)
    n_correct = sum(1 for r in results if r.get("answer_correct", False))
    n_valid_box = sum(1 for r in results if r.get("box_valid", False))
    n_parsed = sum(1 for r in results if r.get("parsed_answer") is not None)
    n_rsa = sum(1 for r in results if r.get("rsa") is not None)

    print("\n" + "=" * 52)
    print("  Visual CoT — Summary")
    print("=" * 52)
    print(f"  Examples          : {n}")
    print(f"  Parsed answers    : {n_parsed} / {n}")
    print(f"  Final Answer Acc  : {faa:.3f}   ({n_correct}/{n} correct)")
    print(f"  Valid box rate    : {vbr:.3f}   ({n_valid_box}/{n} valid)")
    if mrsa is not None:
        print(f"  Mean RSA          : {mrsa:.3f}   (over {n_rsa} examples with targets)")
    else:
        print(f"  Mean RSA          : N/A   (no target boxes available)")
    print("=" * 52)


def print_summary_vcot_verification(results: list[dict]) -> None:
    """Print end-of-run summary for the two-pass Visual CoT + Verification workflow."""
    from metrics import (
        final_answer_accuracy, mean_rsa,
        box_revision_rate, answer_flip_rate, recovery_rate,
    )

    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    # Pass 1 metrics
    p1_correct = sum(1 for r in results if r.get("pass1_answer_correct", False))
    p1_faa = p1_correct / n

    # Pass 2 metrics (use pass2_answer_correct)
    p2_results = [{**r, "answer_correct": r.get("pass2_answer_correct", False)} for r in results]
    p2_faa = final_answer_accuracy(p2_results)

    # RSA (pass1 and pass2)
    p1_rsa_vals = [r["pass1_rsa"] for r in results if r.get("pass1_rsa") is not None]
    p2_rsa_vals = [r["pass2_rsa"] for r in results if r.get("pass2_rsa") is not None]
    p1_rsa = sum(p1_rsa_vals) / len(p1_rsa_vals) if p1_rsa_vals else None
    p2_rsa = sum(p2_rsa_vals) / len(p2_rsa_vals) if p2_rsa_vals else None

    # Verification-specific
    brr = box_revision_rate(results)
    afr = answer_flip_rate(results)
    rec = recovery_rate(results)
    n_wrong_p1 = sum(1 for r in results if not r.get("pass1_answer_correct", True))

    print("\n" + "=" * 56)
    print("  Visual CoT + Verification — Summary")
    print("=" * 56)
    print(f"  Examples              : {n}")
    print(f"  Pass 1 FAA            : {p1_faa:.3f}   ({p1_correct}/{n} correct)")
    print(f"  Pass 2 FAA            : {p2_faa:.3f}   ({int(p2_faa*n)}/{n} correct)")
    if p1_rsa is not None:
        print(f"  Pass 1 RSA            : {p1_rsa:.3f}   (over {len(p1_rsa_vals)} examples)")
    else:
        print(f"  Pass 1 RSA            : N/A")
    if p2_rsa is not None:
        print(f"  Pass 2 RSA            : {p2_rsa:.3f}   (over {len(p2_rsa_vals)} examples)")
    else:
        print(f"  Pass 2 RSA            : N/A")
    print(f"  Box revision rate     : {brr:.3f}   ({int(brr*n)}/{n} revised)")
    print(f"  Answer flip rate      : {afr:.3f}   ({int(afr*n)}/{n} flipped)")
    if rec is not None:
        print(f"  Recovery rate         : {rec:.3f}   ({int(rec*n_wrong_p1)}/{n_wrong_p1} recovered)")
    else:
        print(f"  Recovery rate         : N/A   (pass 1 was always correct)")
    print("=" * 56)


def print_summary_multistage(results: list[dict]) -> None:
    """Print end-of-run summary for the Visual CoT + Multi-Stage Correction pipeline."""
    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    init_correct  = sum(1 for r in results if r.get("initial_answer_correct", False))
    final_correct = sum(1 for r in results if r.get("final_answer_correct", False))
    triggered     = sum(1 for r in results if r.get("correction_triggered", False))
    box_revised   = sum(1 for r in results if r.get("box_revised", False))
    flipped       = sum(1 for r in results if r.get("answer_flipped", False))
    wrong_init    = [r for r in results if not r.get("initial_answer_correct", True)]
    recovered     = sum(1 for r in wrong_init if r.get("final_answer_correct", False))

    bqs_vals  = [r["bqs"] for r in results if r.get("bqs") is not None]
    avg_bqs   = sum(bqs_vals) / len(bqs_vals) if bqs_vals else None

    init_rsa_vals  = [r["initial_rsa"] for r in results if r.get("initial_rsa") is not None]
    final_rsa_vals = [r["revised_rsa"]  for r in results if r.get("revised_rsa")  is not None]
    init_rsa  = sum(init_rsa_vals)  / len(init_rsa_vals)  if init_rsa_vals  else None
    final_rsa = sum(final_rsa_vals) / len(final_rsa_vals) if final_rsa_vals else None

    print("\n" + "=" * 60)
    print("  Visual CoT + Multi-Stage Correction — Summary")
    print("=" * 60)
    print(f"  Examples               : {n}")
    print(f"  Initial FAA            : {init_correct/n:.3f}   ({init_correct}/{n} correct)")
    print(f"  Final FAA              : {final_correct/n:.3f}   ({final_correct}/{n} correct)")
    if init_rsa is not None:
        print(f"  Initial RSA            : {init_rsa:.3f}   (over {len(init_rsa_vals)} examples)")
    else:
        print(f"  Initial RSA            : N/A")
    if final_rsa is not None:
        print(f"  Revised RSA            : {final_rsa:.3f}   (over {len(final_rsa_vals)} examples)")
    else:
        print(f"  Revised RSA            : N/A")
    if avg_bqs is not None:
        print(f"  Mean BQS               : {avg_bqs:.3f}")
    print(f"  Correction trigger rate: {triggered/n:.3f}   ({triggered}/{n} triggered)")
    print(f"  Box revision rate      : {box_revised/n:.3f}   ({box_revised}/{n} revised)")
    print(f"  Answer flip rate       : {flipped/n:.3f}   ({flipped}/{n} flipped)")
    if wrong_init:
        print(f"  Recovery rate          : {recovered/len(wrong_init):.3f}"
              f"   ({recovered}/{len(wrong_init)} recovered)")
    else:
        print(f"  Recovery rate          : N/A   (initial was always correct)")
    print("=" * 60)


def print_summary_verification(results: list[dict]) -> None:
    """Print a short summary table for a Verification run."""
    from metrics import final_answer_accuracy, valid_box_rate, mean_rsa, answer_change_rate

    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    # Verification-pass metrics (uses "answer_correct" which is set from verification answer)
    faa = final_answer_accuracy(results)
    vbr = valid_box_rate(results)
    mrsa = mean_rsa(results)
    acr = answer_change_rate(results)
    n_correct = sum(1 for r in results if r.get("answer_correct", False))
    n_changed = sum(1 for r in results if r.get("answer_changed", False))
    n_parsed = sum(1 for r in results if r.get("parsed_answer") is not None)
    n_rsa = sum(1 for r in results if r.get("rsa") is not None)

    # Initial Visual CoT accuracy for comparison
    n_initial_correct = sum(1 for r in results if r.get("initial_answer_correct", False))
    initial_faa = n_initial_correct / n if n else 0.0

    print("\n" + "=" * 52)
    print("  Visual CoT + Verification — Summary")
    print("=" * 52)
    print(f"  Examples             : {n}")
    print(f"  Parsed answers       : {n_parsed} / {n}")
    print(f"  Initial FAA          : {initial_faa:.3f}   ({n_initial_correct}/{n} correct)")
    print(f"  Verification FAA     : {faa:.3f}   ({n_correct}/{n} correct)")
    print(f"  Answer change rate   : {acr:.3f}   ({n_changed}/{n} changed)")
    print(f"  Valid box rate       : {vbr:.3f}")
    if mrsa is not None:
        print(f"  Mean RSA             : {mrsa:.3f}   (over {n_rsa} examples with targets)")
    else:
        print(f"  Mean RSA             : N/A   (no target boxes available)")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Per-example console log
# ---------------------------------------------------------------------------

def log_example_result(result: dict, mode: str = "textual_cot") -> None:
    """Print one-line status for an example as it completes."""
    eid = result.get("example_id", "?")
    ans = result.get("parsed_answer", "?")
    gt = result.get("ground_truth", "?")
    ok = "✓" if result.get("answer_correct") else "✗"

    line = f"  [{ok}] {eid}  ans={ans}  gt={gt}"

    if mode in ("visual_cot", "verification"):
        bv = result.get("box_valid", False)
        box_status = "box=ok" if bv else f"box=INVALID({result.get('box_invalid_reason','')})"
        iou_val = result.get("iou")
        iou_str = f"  iou={iou_val:.3f}" if iou_val is not None else ""
        line += f"  {box_status}{iou_str}"

    if mode == "verification":
        changed = result.get("answer_changed", False)
        init_ans = result.get("initial_answer", "?")
        line += f"  init={init_ans}  {'CHANGED' if changed else 'same'}"

    print(line)
