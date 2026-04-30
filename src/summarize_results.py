"""
Collect all run results into a single summary CSV.

One row per (method, condition) combination. Loads clean and corrupted results
from the results/ directory and writes results/summary.csv.

Columns:
  method            textual | visual | verification
  condition         clean | blur-1 | blur-3 | ... | rot-90
  corruption_type   clean | blur | noise | rotation
  severity          0 for clean, else the numeric parameter (sigma/variance/degrees)
  faa               Final Answer Accuracy (pass-2 for verification)
  rsa               Mean RSA — blank for textual, blank for rotation (geometry invalid)
  valid_box_rate    Fraction of structurally valid predicted boxes — blank for textual
  box_revision_rate Fraction where pass-2 box differs from pass-1 — verification only
  answer_flip_rate  Fraction where pass-2 answer differs from pass-1 — verification only
  recovery_rate     Among P1-wrong, fraction where P2 is correct — verification only
  n_examples        Number of examples in the run

Usage:
    python src/summarize_results.py
    → results/summary.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corruptions import get_all_corruption_specs

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_PATH = os.path.join(_ROOT, "results", "summary.csv")

CLEAN_PATHS = {
    "textual":      os.path.join(_ROOT, "results", "textual_cot_results.json"),
    "visual":       os.path.join(_ROOT, "results", "visual_cot_results.json"),
    "verification": os.path.join(_ROOT, "results", "visual_cot_verification_results.json"),
    "multistage":   os.path.join(_ROOT, "results", "multistage_correction_results.json"),
}

METHODS = ["textual", "visual", "verification", "multistage"]
_ALL_SPECS = get_all_corruption_specs()
_RSA_VALID = {s.display_name for s in _ALL_SPECS if s.rsa_valid}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load(method: str, condition: str):
    if condition == "clean":
        path = CLEAN_PATHS[method]
    else:
        path = os.path.join(_ROOT, "results", "corrupted", method, f"{condition}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe(val, fmt=".3f"):
    return format(val, fmt) if val is not None else ""


def _metrics(results, method: str, condition: str) -> dict:
    n = len(results)

    # FAA
    if method in ("textual", "visual"):
        correct = sum(1 for r in results if r.get("answer_correct", False))
    elif method == "verification":
        correct = sum(1 for r in results if r.get("pass2_answer_correct", False))
    else:  # multistage
        correct = sum(1 for r in results if r.get("final_answer_correct", False))
    faa = correct / n

    # RSA — only for visual/verification/multistage and geometry-preserving corruptions
    rsa = None
    if method != "textual" and (condition == "clean" or condition in _RSA_VALID):
        if method == "visual":
            key = "rsa"
            vals = [r[key] for r in results if r.get(key) is not None]
        elif method == "verification":
            key = "pass2_rsa"
            vals = [r[key] for r in results if r.get(key) is not None]
        else:  # multistage: prefer revised_rsa, fall back to initial_rsa
            vals = [r["revised_rsa"] for r in results if r.get("revised_rsa") is not None]
            if not vals:
                vals = [r["initial_rsa"] for r in results if r.get("initial_rsa") is not None]
        rsa = sum(vals) / len(vals) if vals else None

    # Valid box rate
    vbr = None
    if method != "textual":
        if method == "visual":
            key = "box_valid"
        elif method == "verification":
            key = "pass2_box_valid"
        else:  # multistage
            key = "initial_box_valid"
        vbr = sum(1 for r in results if r.get(key, False)) / n

    # Correction/verification dynamics (verification + multistage)
    brr = afr = rec = None
    if method == "verification":
        brr = sum(1 for r in results if r.get("box_revised", False)) / n
        afr = sum(1 for r in results if r.get("answer_flipped", False)) / n
        wrong = [r for r in results if not r.get("pass1_answer_correct", True)]
        if wrong:
            rec = sum(1 for r in wrong if r.get("pass2_answer_correct", False)) / len(wrong)
    elif method == "multistage":
        brr = sum(1 for r in results if r.get("box_revised", False)) / n
        afr = sum(1 for r in results if r.get("answer_flipped", False)) / n
        wrong = [r for r in results if not r.get("initial_answer_correct", True)]
        if wrong:
            rec = sum(1 for r in wrong if r.get("final_answer_correct", False)) / len(wrong)

    return {
        "faa":               faa,
        "rsa":               rsa,
        "valid_box_rate":    vbr,
        "box_revision_rate": brr,
        "answer_flip_rate":  afr,
        "recovery_rate":     rec,
        "n_examples":        n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conditions = ["clean"] + [s.display_name for s in _ALL_SPECS]
    condition_meta = {"clean": ("clean", 0)}
    for s in _ALL_SPECS:
        condition_meta[s.display_name] = (s.corruption_type, s.severity)

    rows = []
    missing = 0

    for method in METHODS:
        for condition in conditions:
            results = _load(method, condition)
            if results is None:
                missing += 1
                continue

            m = _metrics(results, method, condition)
            ctype, severity = condition_meta[condition]

            rows.append({
                "method":            method,
                "condition":         condition,
                "corruption_type":   ctype,
                "severity":          severity,
                "faa":               _safe(m["faa"]),
                "rsa":               _safe(m["rsa"]),
                "valid_box_rate":    _safe(m["valid_box_rate"]),
                "box_revision_rate": _safe(m["box_revision_rate"]),
                "answer_flip_rate":  _safe(m["answer_flip_rate"]),
                "recovery_rate":     _safe(m["recovery_rate"]),
                "n_examples":        m["n_examples"],
            })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = [
        "method", "condition", "corruption_type", "severity",
        "faa", "rsa", "valid_box_rate",
        "box_revision_rate", "answer_flip_rate", "recovery_rate",
        "n_examples",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {OUTPUT_PATH}")
    if missing:
        print(f"  ({missing} result files missing — skipped)")

    # Print a quick preview
    print(f"\n{'method':<14} {'condition':<12} {'FAA':>6} {'RSA':>6} {'VBR':>6}")
    print("-" * 48)
    for row in rows:
        print(f"{row['method']:<14} {row['condition']:<12} {row['faa']:>6} {row['rsa']:>6} {row['valid_box_rate']:>6}")


if __name__ == "__main__":
    main()
