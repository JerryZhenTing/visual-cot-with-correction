"""
Aggregate per-example results into paper-ready metric tables.

Reads all result files from results/raw/ (produced by eval_runner.py) and
computes the full set of paper metrics per (model, method) row.

Metric definitions:
    FAA               — Final Answer Accuracy (method-specific answer field)
    RSA               — Mean Region Sensitive Accuracy (IoU >= 0.5; None for rotation)
    Blur FAA          — Mean FAA over blur-{1,3,5}
    Noise FAA         — Mean FAA over noise-{001,005,010}
    Rot. FAA          — Mean FAA over rot-{15,45,90}
    PDR (blur)        — (clean_FAA - blur_FAA) / clean_FAA
    PDR (noise)       — (clean_FAA - noise_FAA) / clean_FAA
    PDR (rotation)    — (clean_FAA - rot_FAA) / clean_FAA
    Box Revision Rate — fraction of examples where pass2 box differs from pass1
                        (verification + multistage only)
    Answer Flip Rate  — fraction of examples where final answer differs from initial
    Recovery Rate     — among pass1-wrong examples, fraction where pass2 is correct

RSA notes:
    RSA is only reported for blur and noise conditions.
    Rotation is excluded because bounding boxes are not re-projected under rotation.
    Clean RSA requires target_box; if none available the value will be null.

Output files:
    results/aggregated/summary.json      — full nested dict
    results/aggregated/paper_table.csv   — one row per (model, method)
    results/aggregated/per_condition.csv — one row per (model, method, condition)

Usage:
    python src/aggregate_results.py
    python src/aggregate_results.py --raw-dir results/raw --out-dir results/aggregated
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import performance_drop_rate

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Corruption families used for family averages and PDR
CORRUPTION_FAMILIES = {
    "blur":     ["blur-1",    "blur-3",    "blur-5"],
    "noise":    ["noise-001", "noise-005", "noise-010"],
    "rotation": ["rot-15",    "rot-45",    "rot-90"],
}

# Which conditions support RSA (geometry preserved → reference boxes still valid)
RSA_VALID_CONDITIONS = {
    "clean", "blur-1", "blur-3", "blur-5",
    "noise-001", "noise-005", "noise-010",
}

METHODS   = ["textual", "visual", "verification", "multistage"]
ALL_CONDITIONS = (
    ["clean"]
    + CORRUPTION_FAMILIES["blur"]
    + CORRUPTION_FAMILIES["noise"]
    + CORRUPTION_FAMILIES["rotation"]
)


# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------

def _parse_filename(fname: str) -> Optional[tuple[str, str, str]]:
    """
    Extract (model, method, condition) from a result filename.
    Expected pattern: {model}_{method}_{condition}.json
    Conditions with hyphens like 'blur-1' are handled.
    Returns None if the name does not match.
    """
    name = fname.removesuffix(".json")
    # Try splitting on first two underscores; condition may contain hyphens
    parts = name.split("_", 2)
    if len(parts) != 3:
        return None
    model, method, condition = parts
    from eval_config import VALID_MODELS, VALID_METHODS, VALID_CONDITIONS
    if model not in VALID_MODELS:
        return None
    if method not in VALID_METHODS:
        return None
    if condition not in VALID_CONDITIONS:
        return None
    return model, method, condition


def scan_raw_results(raw_dir: str) -> dict[tuple[str, str, str], str]:
    """
    Discover all result files in raw_dir.
    Returns {(model, method, condition): filepath}.
    """
    index: dict[tuple[str, str, str], str] = {}
    if not os.path.isdir(raw_dir):
        return index
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        index[parsed] = os.path.join(raw_dir, fname)
    return index


def load_results(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-condition metric extraction
# ---------------------------------------------------------------------------

def _faa(results: list[dict], method: str) -> Optional[float]:
    """Final Answer Accuracy — field name depends on method."""
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


def _rsa(results: list[dict], method: str, condition: str) -> Optional[float]:
    """Mean RSA — None for textual or rotation conditions."""
    if method == "textual":
        return None
    if condition not in RSA_VALID_CONDITIONS:
        return None  # rotation: boxes not valid

    if method == "visual":
        vals = [r["rsa"] for r in results if r.get("rsa") is not None]
    elif method == "verification":
        vals = [r["pass2_rsa"] for r in results if r.get("pass2_rsa") is not None]
    else:  # multistage: prefer revised_rsa, fall back to initial_rsa
        vals = [r["revised_rsa"] for r in results if r.get("revised_rsa") is not None]
        if not vals:
            vals = [r["initial_rsa"] for r in results if r.get("initial_rsa") is not None]

    return sum(vals) / len(vals) if vals else None


def _brr(results: list[dict], method: str) -> Optional[float]:
    """Box Revision Rate — verification and multistage only."""
    if method not in ("verification", "multistage"):
        return None
    n = len(results)
    return sum(1 for r in results if r.get("box_revised", False)) / n if n else None


def _afr(results: list[dict], method: str) -> Optional[float]:
    """Answer Flip Rate — verification and multistage only."""
    if method not in ("verification", "multistage"):
        return None
    n = len(results)
    return sum(1 for r in results if r.get("answer_flipped", False)) / n if n else None


def _recovery(results: list[dict], method: str) -> Optional[float]:
    """Recovery Rate — among pass1-wrong, fraction where final answer is correct."""
    if method not in ("verification", "multistage"):
        return None
    if method == "verification":
        wrong_key, correct_key = "pass1_answer_correct", "pass2_answer_correct"
    else:
        wrong_key, correct_key = "initial_answer_correct", "final_answer_correct"
    wrong = [r for r in results if not r.get(wrong_key, True)]
    if not wrong:
        return None
    return sum(1 for r in wrong if r.get(correct_key, False)) / len(wrong)


def _vbr(results: list[dict], method: str) -> Optional[float]:
    """Valid Box Rate."""
    if method == "textual":
        return None
    n = len(results)
    if not n:
        return None
    key = (
        "box_valid"      if method == "visual"
        else "pass2_box_valid"  if method == "verification"
        else "initial_box_valid"
    )
    return sum(1 for r in results if r.get(key, False)) / n


def _mean_bqs(results: list[dict], method: str) -> Optional[float]:
    """Mean BQS — multistage only."""
    if method != "multistage":
        return None
    vals = [r["bqs"] for r in results if r.get("bqs") is not None]
    return sum(vals) / len(vals) if vals else None


def compute_condition_metrics(
    results: list[dict], method: str, condition: str
) -> dict:
    """Compute all metrics for a single (method, condition) result set."""
    return {
        "n":               len(results),
        "faa":             _faa(results, method),
        "rsa":             _rsa(results, method, condition),
        "vbr":             _vbr(results, method),
        "brr":             _brr(results, method),
        "afr":             _afr(results, method),
        "recovery":        _recovery(results, method),
        "mean_bqs":        _mean_bqs(results, method),
    }


# ---------------------------------------------------------------------------
# Full model × method aggregation
# ---------------------------------------------------------------------------

def aggregate_all(raw_dir: str) -> dict:
    """
    Load all results and compute metrics.
    Returns nested dict:
        {model: {method: {condition: metrics_dict}}}
    """
    index = scan_raw_results(raw_dir)

    data: dict = {}
    for (model, method, condition), filepath in index.items():
        results  = load_results(filepath)
        metrics  = compute_condition_metrics(results, method, condition)
        data.setdefault(model, {}).setdefault(method, {})[condition] = metrics

    return data


# ---------------------------------------------------------------------------
# Paper table builder
# ---------------------------------------------------------------------------

def build_paper_table(data: dict) -> list[dict]:
    """
    Produce one row per (model, method) with:
      - Clean FAA / RSA
      - Blur / Noise / Rotation family-average FAA and RSA
      - PDR per family
      - BRR, AFR, Recovery (verification + multistage only)
    """
    rows = []

    for model, methods in data.items():
        for method, conditions in methods.items():
            row: dict = {"model": model, "method": method}

            # Clean
            clean = conditions.get("clean", {})
            row["clean_faa"] = clean.get("faa")
            row["clean_rsa"] = clean.get("rsa")
            row["clean_n"]   = clean.get("n")

            # Family averages
            for family, members in CORRUPTION_FAMILIES.items():
                faas = [conditions[c]["faa"]
                        for c in members
                        if c in conditions and conditions[c].get("faa") is not None]
                rsas = [conditions[c]["rsa"]
                        for c in members
                        if c in conditions and conditions[c].get("rsa") is not None]
                n_family = sum(conditions[c].get("n", 0)
                               for c in members if c in conditions)

                row[f"{family}_faa"] = sum(faas) / len(faas) if faas else None
                row[f"{family}_rsa"] = sum(rsas) / len(rsas) if rsas else None
                row[f"{family}_n"]   = n_family if n_family else None

                # PDR: rotation RSA is excluded by design
                row[f"{family}_pdr"] = performance_drop_rate(
                    clean_faa=row["clean_faa"], corrupted_faas=faas
                )

            # Verification dynamics (clean condition as representative)
            # We average across all available conditions for a stable estimate
            all_results_list = []  # unused — just use clean for dynamics
            row["brr"]      = clean.get("brr")
            row["afr"]      = clean.get("afr")
            row["recovery"] = clean.get("recovery")
            row["mean_bqs"] = clean.get("mean_bqs")

            rows.append(row)

    return rows


def build_per_condition_table(data: dict) -> list[dict]:
    """One row per (model, method, condition) — for supplementary tables."""
    rows = []
    for model, methods in data.items():
        for method, conditions in methods.items():
            for condition in ALL_CONDITIONS:
                if condition not in conditions:
                    continue
                m = conditions[condition]
                rows.append({
                    "model":     model,
                    "method":    method,
                    "condition": condition,
                    "n":         m.get("n"),
                    "faa":       m.get("faa"),
                    "rsa":       m.get("rsa"),
                    "vbr":       m.get("vbr"),
                    "brr":       m.get("brr"),
                    "afr":       m.get("afr"),
                    "recovery":  m.get("recovery"),
                    "mean_bqs":  m.get("mean_bqs"),
                })
    return rows


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: Optional[float], decimals: int = 1) -> str:
    """Format a [0,1] fraction as a percentage string, or '---' if None."""
    if v is None:
        return "---"
    return f"{v * 100:.{decimals}f}%"


def _fmt(v: Optional[float], decimals: int = 3) -> str:
    """Format a float or return '---'."""
    if v is None:
        return "---"
    return f"{v:.{decimals}f}"


def _csv_val(v) -> str:
    """Render a value for CSV output."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(data: dict) -> None:
    rows = build_paper_table(data)
    if not rows:
        print("No results found in data.")
        return

    header = (
        f"{'Model':<8} {'Method':<14} {'Clean':>7} {'Blur':>7} "
        f"{'Noise':>7} {'Rot':>7} {'PDR-B':>7} {'PDR-N':>7} {'PDR-R':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  Aggregated Results (FAA)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['model']:<8} {r['method']:<14} "
            f"{_pct(r.get('clean_faa')):>7} "
            f"{_pct(r.get('blur_faa')):>7} "
            f"{_pct(r.get('noise_faa')):>7} "
            f"{_pct(r.get('rotation_faa')):>7} "
            f"{_pct(r.get('blur_pdr')):>7} "
            f"{_pct(r.get('noise_pdr')):>7} "
            f"{_pct(r.get('rotation_pdr')):>7}"
        )
    print(sep)

    # Verification / multistage dynamics
    dyn_rows = [r for r in rows if r["method"] in ("verification", "multistage")]
    if dyn_rows:
        print(f"\n{'Model':<8} {'Method':<14} {'BRR':>7} {'AFR':>7} {'Recov':>7} {'BQS':>7}")
        print(sep)
        for r in dyn_rows:
            print(
                f"{r['model']:<8} {r['method']:<14} "
                f"{_pct(r.get('brr')):>7} "
                f"{_pct(r.get('afr')):>7} "
                f"{_pct(r.get('recovery')):>7} "
                f"{_pct(r.get('mean_bqs')):>7}"
            )
        print(sep)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Wrote {path}")


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_val(v) for k, v in row.items()})
    print(f"  Wrote {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(raw_dir: Optional[str] = None, out_dir: Optional[str] = None) -> None:
    if raw_dir is None:
        raw_dir = os.path.join(_ROOT, "results", "raw")
    if out_dir is None:
        out_dir = os.path.join(_ROOT, "results", "aggregated")

    print(f"Scanning {raw_dir} ...")
    index = scan_raw_results(raw_dir)
    if not index:
        print("No result files found. Run eval_runner.py first.")
        return
    print(f"  Found {len(index)} result files.")

    data = aggregate_all(raw_dir)
    print_summary(data)

    write_json(data, os.path.join(out_dir, "summary.json"))
    write_csv(build_paper_table(data),         os.path.join(out_dir, "paper_table.csv"))
    write_csv(build_per_condition_table(data), os.path.join(out_dir, "per_condition.csv"))

    print(f"\nDone. Aggregated outputs in {out_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Aggregate raw results into paper tables.")
    p.add_argument("--raw-dir", default=None,
                   help="Directory with per-example JSON files (default: results/raw)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for tables (default: results/aggregated)")
    args = p.parse_args()
    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
