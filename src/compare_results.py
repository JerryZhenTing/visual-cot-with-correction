"""
Compare FAA, RSA, and valid box rate across all three inference settings.

Loads:
  results/textual_cot_results.json            (Setting 1)
  results/visual_cot_results.json             (Setting 2)
  results/visual_cot_verification_results.json (Setting 3 — pass 2 is the final answer)

Prints a side-by-side table.  Missing result files are reported but do not
crash the script.

Usage:
    cd /path/to/project
    python src/compare_results.py
"""

from __future__ import annotations

import json
import os
from typing import Optional

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

RESULT_FILES = {
    "Textual CoT (S1)":        os.path.join(_ROOT, "results", "textual_cot_results.json"),
    "Visual CoT (S2)":         os.path.join(_ROOT, "results", "visual_cot_results.json"),
    "VCoT+Verify (S3)":        os.path.join(_ROOT, "results", "visual_cot_verification_results.json"),
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load(path: str) -> Optional[list[dict]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-setting metric extractors
# ---------------------------------------------------------------------------

def _metrics_textual(results: list[dict]) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r.get("answer_correct", False))
    parsed  = sum(1 for r in results if r.get("parsed_answer") is not None)
    return {
        "n":            n,
        "faa":          correct / n if n else 0.0,
        "parsed_rate":  parsed  / n if n else 0.0,
        "valid_box_rate": None,
        "mean_rsa":     None,
        "pass1_faa":    None,
    }


def _metrics_visual(results: list[dict]) -> dict:
    n = len(results)
    correct   = sum(1 for r in results if r.get("answer_correct", False))
    parsed    = sum(1 for r in results if r.get("parsed_answer") is not None)
    valid_box = sum(1 for r in results if r.get("box_valid", False))
    rsa_vals  = [r["rsa"] for r in results if r.get("rsa") is not None]
    return {
        "n":              n,
        "faa":            correct / n if n else 0.0,
        "parsed_rate":    parsed  / n if n else 0.0,
        "valid_box_rate": valid_box / n if n else 0.0,
        "mean_rsa":       sum(rsa_vals) / len(rsa_vals) if rsa_vals else None,
        "pass1_faa":      None,
    }


def _metrics_verification(results: list[dict]) -> dict:
    """Setting 3: pass-2 answer is the final answer; also report pass-1 FAA."""
    n = len(results)
    # Pass 1
    p1_correct = sum(1 for r in results if r.get("pass1_answer_correct", False))
    # Pass 2 (final)
    p2_correct = sum(1 for r in results if r.get("pass2_answer_correct", False))
    # Box quality: use pass-2 box_valid
    valid_box = sum(1 for r in results if r.get("pass2_box_valid", False))
    # RSA: pass-2
    rsa_vals = [r["pass2_rsa"] for r in results if r.get("pass2_rsa") is not None]
    # Verification-specific
    box_revised   = sum(1 for r in results if r.get("box_revised", False))
    ans_flipped   = sum(1 for r in results if r.get("answer_flipped", False))
    wrong_p1      = [r for r in results if not r.get("pass1_answer_correct", True)]
    recovered     = sum(1 for r in wrong_p1 if r.get("pass2_answer_correct", False))
    return {
        "n":                 n,
        "faa":               p2_correct / n if n else 0.0,   # final answer = pass 2
        "parsed_rate":       None,
        "valid_box_rate":    valid_box / n if n else 0.0,
        "mean_rsa":          sum(rsa_vals) / len(rsa_vals) if rsa_vals else None,
        "pass1_faa":         p1_correct / n if n else 0.0,
        "box_revised_rate":  box_revised / n if n else 0.0,
        "answer_flip_rate":  ans_flipped / n if n else 0.0,
        "recovery_rate":     recovered / len(wrong_p1) if wrong_p1 else None,
    }


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def _fmt(val, pct: bool = True, width: int = 8) -> str:
    if val is None:
        return " " * (width - 3) + "N/A"
    if pct:
        return f"{val * 100:>{width-1}.1f}%"
    return f"{val:>{width}.1f}"


def print_comparison(stats: dict[str, dict]) -> None:
    names   = list(stats.keys())
    COL     = 18   # column width

    hdr_line  = "  {:<28}".format("Metric")
    sep_line  = "  " + "-" * 28
    for name in names:
        hdr_line += f"  {name:>{COL}}"
        sep_line += "  " + "-" * COL

    print("\n" + "=" * (30 + (COL + 2) * len(names)))
    print("  Comparison: Visual CoT Robustness Settings")
    print("=" * (30 + (COL + 2) * len(names)))
    print(hdr_line)
    print(sep_line)

    rows = [
        ("Examples (n)",          "n",                False),
        ("Final Answer Acc (FAA)","faa",               True),
        ("  Pass-1 FAA",          "pass1_faa",         True),
        ("Valid Box Rate",        "valid_box_rate",    True),
        ("Mean RSA (IoU≥0.5)",    "mean_rsa",          True),
        ("Box Revision Rate",     "box_revised_rate",  True),
        ("Answer Flip Rate",      "answer_flip_rate",  True),
        ("Recovery Rate",         "recovery_rate",     True),
    ]

    for label, key, pct in rows:
        row = f"  {label:<28}"
        for name in names:
            val = stats[name].get(key)
            if key == "n":
                cell = f"{val:>{COL}}" if val is not None else " " * (COL - 3) + "N/A"
            else:
                raw = _fmt(val, pct=pct, width=COL)
                cell = raw
            row += f"  {cell}"
        print(row)

    print("=" * (30 + (COL + 2) * len(names)))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stats: dict[str, dict] = {}
    dispatchers = {
        "Textual CoT (S1)": _metrics_textual,
        "Visual CoT (S2)":  _metrics_visual,
        "VCoT+Verify (S3)": _metrics_verification,
    }

    for name, path in RESULT_FILES.items():
        results = _load(path)
        if results is None:
            print(f"  [MISSING] {name}: {os.path.relpath(path, _ROOT)}")
            stats[name] = {
                "n": None, "faa": None, "pass1_faa": None,
                "valid_box_rate": None, "mean_rsa": None,
                "box_revised_rate": None, "answer_flip_rate": None,
                "recovery_rate": None,
            }
        else:
            stats[name] = dispatchers[name](results)
            print(f"  [OK]      {name}: {len(results)} examples — {os.path.relpath(path, _ROOT)}")

    print_comparison(stats)


if __name__ == "__main__":
    main()
