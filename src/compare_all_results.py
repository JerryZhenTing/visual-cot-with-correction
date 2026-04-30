"""
Full comparison table: FAA, RSA, and verification metrics across all methods
and all conditions (clean + 9 corruptions).

Loads:
  results/textual_cot_results.json               ← clean, method=textual
  results/visual_cot_results.json                ← clean, method=visual
  results/visual_cot_verification_results.json   ← clean, method=verification
  results/corrupted/<method>/<condition>.json    ← corrupted runs

Prints four tables:
  1. FAA        — all methods × all conditions
  2. RSA        — visual + verification × blur/noise only
  3. Valid Box  — visual + verification × all conditions
  4. Verify extras — box revision, answer flip, recovery × all conditions

Missing files are shown as "---" rather than crashing.
RSA columns for rotation conditions are shown as "N/A" (geometry invalid).

Usage:
    cd /path/to/project
    python src/compare_all_results.py
"""

from __future__ import annotations

import json
import os
from typing import Optional

from corruptions import get_all_corruption_specs

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Condition ordering
_ALL_SPECS    = get_all_corruption_specs()
CORRUPTIONS   = [s.display_name for s in _ALL_SPECS]          # 9
CONDITIONS    = ["clean"] + CORRUPTIONS                         # 10
RSA_CONDITIONS = ["clean"] + [s.display_name for s in _ALL_SPECS if s.rsa_valid]  # 7
_RSA_VALID_SET = {s.display_name for s in _ALL_SPECS if s.rsa_valid}

CLEAN_PATHS = {
    "textual":      os.path.join(_ROOT, "results", "textual_cot_results.json"),
    "visual":       os.path.join(_ROOT, "results", "visual_cot_results.json"),
    "verification": os.path.join(_ROOT, "results", "visual_cot_verification_results.json"),
    "multistage":   os.path.join(_ROOT, "results", "multistage_correction_results.json"),
}

METHODS       = ["textual", "visual", "verification", "multistage"]
METHOD_LABELS = {
    "textual":      "Textual CoT",
    "visual":       "Visual CoT",
    "verification": "VCoT+Verify",
    "multistage":   "MultiStage",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load(method: str, condition: str) -> Optional[list[dict]]:
    if condition == "clean":
        path = CLEAN_PATHS[method]
    else:
        path = os.path.join(_ROOT, "results", "corrupted", method, f"{condition}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric extractors  (return None if data unavailable)
# ---------------------------------------------------------------------------

def _faa(results: list[dict], method: str) -> Optional[float]:
    n = len(results)
    if not n:
        return None
    if method in ("textual", "visual"):
        c = sum(1 for r in results if r.get("answer_correct", False))
    elif method == "verification":
        c = sum(1 for r in results if r.get("pass2_answer_correct", False))
    else:  # multistage
        c = sum(1 for r in results if r.get("final_answer_correct", False))
    return c / n


def _rsa(results: list[dict], method: str) -> Optional[float]:
    """Mean RSA — None if no RSA values present (rotation or missing boxes)."""
    if method == "textual":
        return None
    if method == "visual":
        key = "rsa"
    elif method == "verification":
        key = "pass2_rsa"
    else:  # multistage: use revised_rsa (box after correction), fall back to initial_rsa
        vals = [r["revised_rsa"] for r in results if r.get("revised_rsa") is not None]
        if not vals:
            vals = [r["initial_rsa"] for r in results if r.get("initial_rsa") is not None]
        return sum(vals) / len(vals) if vals else None
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _vbr(results: list[dict], method: str) -> Optional[float]:
    """Valid box rate."""
    if method == "textual":
        return None
    n = len(results)
    if not n:
        return None
    if method == "visual":
        key = "box_valid"
    elif method == "verification":
        key = "pass2_box_valid"
    else:  # multistage: use initial box validity
        key = "initial_box_valid"
    v = sum(1 for r in results if r.get(key, False))
    return v / n


def _box_revision_rate(results: list[dict]) -> Optional[float]:
    n = len(results)
    return sum(1 for r in results if r.get("box_revised", False)) / n if n else None


def _answer_flip_rate(results: list[dict]) -> Optional[float]:
    n = len(results)
    return sum(1 for r in results if r.get("answer_flipped", False)) / n if n else None


def _recovery_rate(results: list[dict], method: str) -> Optional[float]:
    if method == "verification":
        wrong = [r for r in results if not r.get("pass1_answer_correct", True)]
        if not wrong:
            return None
        return sum(1 for r in wrong if r.get("pass2_answer_correct", False)) / len(wrong)
    else:  # multistage
        wrong = [r for r in results if not r.get("initial_answer_correct", True)]
        if not wrong:
            return None
        return sum(1 for r in wrong if r.get("final_answer_correct", False)) / len(wrong)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_MISS = "  ---"    # result file missing
_NA   = "  N/A"   # metric not applicable for this corruption type

def _pct(val: Optional[float], width: int = 7) -> str:
    if val is None:
        return _MISS.rjust(width)
    return f"{val*100:>{width}.1f}%"


def _build_table(title: str, col_labels: list[str], rows: list[tuple]) -> str:
    """
    rows: list of (row_label, [cell_str, ...])
    cells are already formatted strings (use _pct or custom).
    """
    cw   = max(len(c) for c in col_labels) + 2   # column width
    lcw  = max(len(r[0]) for r in rows) + 2       # label column width
    lcw  = max(lcw, 16)

    sep   = "=" * (lcw + len(col_labels) * (cw + 2) + 2)
    hdiv  = "-" * len(sep)

    lines = [sep, f"  {title}", sep]
    header = " " * (lcw + 2) + "".join(f"{c:>{cw+2}}" for c in col_labels)
    lines.append(header)
    lines.append(hdiv)

    for label, cells in rows:
        row = f"  {label:<{lcw}}" + "".join(f"{c:>{cw+2}}" for c in cells)
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def _gather() -> dict:
    """
    Returns nested dict: data[method][condition] = results list or None.
    Also prints a load status line per file.
    """
    data: dict[str, dict[str, Optional[list]]] = {}
    ok = missing = 0
    for method in METHODS:
        data[method] = {}
        for cond in CONDITIONS:
            results = _load(method, cond)
            data[method][cond] = results
            if results is None:
                missing += 1
            else:
                ok += 1
    print(f"  Loaded {ok} result files  ({missing} missing)\n")
    return data


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------

def _table_faa(data: dict) -> str:
    col_labels = [METHOD_LABELS[m] for m in METHODS]
    rows = []
    for cond in CONDITIONS:
        cells = []
        for method in METHODS:
            results = data[method][cond]
            if results is None:
                cells.append(_MISS)
            else:
                cells.append(_pct(_faa(results, method)))
        rows.append((cond, cells))
    return _build_table("Final Answer Accuracy (FAA)", col_labels, rows)


def _table_rsa(data: dict) -> str:
    vis_methods = ["visual", "verification", "multistage"]
    col_labels  = [METHOD_LABELS[m] for m in vis_methods]
    rows = []
    for cond in RSA_CONDITIONS:
        cells = []
        for method in vis_methods:
            results = data[method][cond]
            if results is None:
                cells.append(_MISS)
            elif cond != "clean" and cond not in _RSA_VALID_SET:
                cells.append(_NA)
            else:
                val = _rsa(results, method)
                cells.append(_pct(val) if val is not None else "   ---")
        rows.append((cond, cells))
    return _build_table(
        "Mean RSA (IoU \u2265 0.5)  [blur + noise only — rotation omitted]",
        col_labels, rows,
    )


def _table_vbr(data: dict) -> str:
    vis_methods = ["visual", "verification", "multistage"]
    col_labels  = [METHOD_LABELS[m] for m in vis_methods]
    rows = []
    for cond in CONDITIONS:
        cells = []
        for method in vis_methods:
            results = data[method][cond]
            if results is None:
                cells.append(_MISS)
            else:
                cells.append(_pct(_vbr(results, method)))
        rows.append((cond, cells))
    return _build_table("Valid Box Rate", col_labels, rows)


def _table_verify(data: dict) -> str:
    col_labels = ["Box Revised", "Ans Flipped", "Recovery"]
    rows = []
    for cond in CONDITIONS:
        results = data["verification"][cond]
        if results is None:
            cells = [_MISS, _MISS, _MISS]
        else:
            rec = _recovery_rate(results, "verification")
            cells = [
                _pct(_box_revision_rate(results)),
                _pct(_answer_flip_rate(results)),
                _pct(rec) if rec is not None else "  N/A ",
            ]
        rows.append((cond, cells))
    return _build_table(
        "Verification Dynamics  (VCoT+Verify only)",
        col_labels, rows,
    )


def _table_multistage(data: dict) -> str:
    col_labels = ["Box Revised", "Ans Flipped", "Recovery", "Triggered", "Mean BQS"]
    rows = []
    for cond in CONDITIONS:
        results = data["multistage"][cond]
        if results is None:
            cells = [_MISS] * 5
        else:
            n = len(results)
            rec = _recovery_rate(results, "multistage")
            triggered = sum(1 for r in results if r.get("correction_triggered", False))
            bqs_vals  = [r["bqs"] for r in results if r.get("bqs") is not None]
            avg_bqs   = sum(bqs_vals) / len(bqs_vals) if bqs_vals else None
            cells = [
                _pct(_box_revision_rate(results)),
                _pct(_answer_flip_rate(results)),
                _pct(rec) if rec is not None else "  N/A ",
                _pct(triggered / n),
                _pct(avg_bqs) if avg_bqs is not None else _MISS,
            ]
        rows.append((cond, cells))
    return _build_table(
        "Multi-Stage Correction Dynamics  (MultiStage only)",
        col_labels, rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results...\n")
    data = _gather()

    for table_fn in [_table_faa, _table_rsa, _table_vbr, _table_verify, _table_multistage]:
        print(table_fn(data))
        print()


if __name__ == "__main__":
    main()
