"""
Aggregate targeted perturbation results into adversarial-results tables.

Reads all result files from results/targeted_raw/ and computes metrics
per (model, method, perturbation_type, severity), then builds:
  - results/targeted_aggregated/adversarial_summary.csv  (paper table)
  - results/targeted_aggregated/per_spec.csv             (one row per spec)
  - results/targeted_aggregated/summary.json             (full nested dict)

Metrics:
  FAA    — Final Answer Accuracy (method-specific answer field)
  RSA    — Mean Region Sensitive Accuracy (IoU >= 0.5)
  BRR    — Box Revision Rate     (verification + multistage only)
  AFR    — Answer Flip Rate      (verification + multistage only)
  RR     — Recovery Rate         (verification + multistage only)

Paper table columns:
  Model | Method | Occ FAA | Occ RSA | Mask FAA | Mask RSA | Dist FAA | Dist RSA | BRR | AFR | RR

  Family averages: average over both targets (obj1+obj2 for occlusion/distractor)
  and all severities, giving one number per family per (model, method).

Usage:
  python src/aggregate_targeted_results.py
  python src/aggregate_targeted_results.py --raw-dir results/targeted_raw \\
      --out-dir results/targeted_aggregated
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adversarial_perturbations import get_all_targeted_perturbation_specs

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

VALID_MODELS   = ("qwen", "llava")
VALID_METHODS  = ("textual", "visual", "verification", "multistage")
_SPEC_NAMES    = {s.display_name for s in get_all_targeted_perturbation_specs()}
_SPEC_BY_NAME  = {s.display_name: s for s in get_all_targeted_perturbation_specs()}

# Perturbation families for paper table
PERTURBATION_FAMILIES = {
    "occlusion":  [s.display_name for s in get_all_targeted_perturbation_specs()
                   if s.perturbation_type == "occlusion"],
    "mask":       [s.display_name for s in get_all_targeted_perturbation_specs()
                   if s.perturbation_type == "mask"],
    "distractor": [s.display_name for s in get_all_targeted_perturbation_specs()
                   if s.perturbation_type == "distractor"],
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _parse_filename(fname: str) -> Optional[tuple[str, str, str]]:
    """
    Extract (model, method, display_name) from a result filename.
    Pattern: {model}_{method}_{display_name}.json
    Returns None if the name does not match a known targeted perturbation.
    """
    name = fname.removesuffix(".json")
    parts = name.split("_", 2)
    if len(parts) != 3:
        return None
    model, method, display_name = parts
    if model not in VALID_MODELS:
        return None
    if method not in VALID_METHODS:
        return None
    if display_name not in _SPEC_NAMES:
        return None
    return model, method, display_name


def scan_targeted_results(raw_dir: str) -> dict[tuple[str, str, str], str]:
    """Return {(model, method, display_name): filepath}."""
    index: dict[tuple[str, str, str], str] = {}
    if not os.path.isdir(raw_dir):
        return index
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        parsed = _parse_filename(fname)
        if parsed:
            index[parsed] = os.path.join(raw_dir, fname)
    return index


# ---------------------------------------------------------------------------
# Per-spec metric extraction (matches aggregate_results.py conventions)
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
    if method == "textual":
        return None
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
    if method not in ("verification", "multistage"):
        return None
    n = len(results)
    return sum(1 for r in results if r.get("box_revised", False)) / n if n else None


def _afr(results: list[dict], method: str) -> Optional[float]:
    if method not in ("verification", "multistage"):
        return None
    n = len(results)
    return sum(1 for r in results if r.get("answer_flipped", False)) / n if n else None


def _rr(results: list[dict], method: str) -> Optional[float]:
    """Recovery Rate: among pass1-wrong, fraction where final is correct."""
    if method not in ("verification", "multistage"):
        return None
    if method == "verification":
        wrong_key, right_key = "pass1_answer_correct", "pass2_answer_correct"
    else:
        wrong_key, right_key = "initial_answer_correct", "final_answer_correct"
    wrong = [r for r in results if not r.get(wrong_key, True)]
    if not wrong:
        return None
    return sum(1 for r in wrong if r.get(right_key, False)) / len(wrong)


def compute_spec_metrics(results: list[dict], method: str) -> dict:
    return {
        "n":   len(results),
        "faa": _faa(results, method),
        "rsa": _rsa(results, method),
        "brr": _brr(results, method),
        "afr": _afr(results, method),
        "rr":  _rr(results, method),
    }


# ---------------------------------------------------------------------------
# Full aggregation
# ---------------------------------------------------------------------------

def aggregate_all(raw_dir: str) -> dict:
    """
    Load all targeted result files and compute metrics.
    Returns: {model: {method: {display_name: metrics_dict}}}
    """
    index = scan_targeted_results(raw_dir)
    data: dict = {}
    for (model, method, display_name), filepath in index.items():
        with open(filepath) as f:
            results = json.load(f)
        metrics = compute_spec_metrics(results, method)
        data.setdefault(model, {}).setdefault(method, {})[display_name] = metrics
    return data


# ---------------------------------------------------------------------------
# Paper table
# ---------------------------------------------------------------------------

def build_paper_table(data: dict) -> list[dict]:
    """
    One row per (model, method) with family-averaged FAA and RSA.
    Families: occlusion (obj1+obj2), mask (union), distractor (near_obj1+obj2).
    """
    rows = []
    for model, methods in data.items():
        for method, specs in methods.items():
            row: dict = {"model": model, "method": method}

            for family, members in PERTURBATION_FAMILIES.items():
                faas = [specs[m]["faa"] for m in members
                        if m in specs and specs[m].get("faa") is not None]
                rsas = [specs[m]["rsa"] for m in members
                        if m in specs and specs[m].get("rsa") is not None]
                ns   = sum(specs[m].get("n", 0) for m in members if m in specs)

                row[f"{family}_faa"] = sum(faas) / len(faas) if faas else None
                row[f"{family}_rsa"] = sum(rsas) / len(rsas) if rsas else None
                row[f"{family}_n"]   = ns or None

            # Dynamics: average across all specs that have them
            all_brrs = [specs[m]["brr"] for m in specs if specs[m].get("brr") is not None]
            all_afrs = [specs[m]["afr"] for m in specs if specs[m].get("afr") is not None]
            all_rrs  = [specs[m]["rr"]  for m in specs if specs[m].get("rr")  is not None]
            row["brr"] = sum(all_brrs) / len(all_brrs) if all_brrs else None
            row["afr"] = sum(all_afrs) / len(all_afrs) if all_afrs else None
            row["rr"]  = sum(all_rrs)  / len(all_rrs)  if all_rrs  else None

            rows.append(row)
    return rows


def build_per_spec_table(data: dict) -> list[dict]:
    """One row per (model, method, display_name)."""
    specs_ordered = [s.display_name for s in get_all_targeted_perturbation_specs()]
    rows = []
    for model, methods in data.items():
        for method, specs in methods.items():
            for dn in specs_ordered:
                if dn not in specs:
                    continue
                m = specs[dn]
                spec_obj = _SPEC_BY_NAME[dn]
                rows.append({
                    "model":            model,
                    "method":           method,
                    "display_name":     dn,
                    "perturbation_type": spec_obj.perturbation_type,
                    "target":           spec_obj.target,
                    "severity":         spec_obj.severity,
                    "n":                m.get("n"),
                    "faa":              m.get("faa"),
                    "rsa":              m.get("rsa"),
                    "brr":              m.get("brr"),
                    "afr":              m.get("afr"),
                    "rr":               m.get("rr"),
                })
    return rows


# ---------------------------------------------------------------------------
# Formatting and output
# ---------------------------------------------------------------------------

def _pct(v: Optional[float], d: int = 1) -> str:
    return "---" if v is None else f"{v * 100:.{d}f}%"


def _csv_val(v) -> str:
    if v is None: return ""
    if isinstance(v, float): return f"{v:.6f}"
    return str(v)


def print_summary(data: dict) -> None:
    rows = build_paper_table(data)
    if not rows:
        print("No targeted results found.")
        return

    header = (f"{'Model':<8} {'Method':<14} "
              f"{'OccFAA':>8} {'OccRSA':>8} "
              f"{'MaskFAA':>8} {'MaskRSA':>8} "
              f"{'DistFAA':>8} {'DistRSA':>8} "
              f"{'BRR':>7} {'AFR':>7} {'RR':>7}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  Targeted Perturbation Results (FAA)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['model']:<8} {r['method']:<14} "
            f"{_pct(r.get('occlusion_faa')):>8} {_pct(r.get('occlusion_rsa')):>8} "
            f"{_pct(r.get('mask_faa')):>8} {_pct(r.get('mask_rsa')):>8} "
            f"{_pct(r.get('distractor_faa')):>8} {_pct(r.get('distractor_rsa')):>8} "
            f"{_pct(r.get('brr')):>7} {_pct(r.get('afr')):>7} {_pct(r.get('rr')):>7}"
        )
    print(sep)


def write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Wrote {path}")


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_val(v) for k, v in row.items()})
    print(f"  Wrote {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(raw_dir: Optional[str] = None, out_dir: Optional[str] = None) -> None:
    if raw_dir is None:
        raw_dir = os.path.join(_ROOT, "results", "targeted_raw")
    if out_dir is None:
        out_dir = os.path.join(_ROOT, "results", "targeted_aggregated")

    print(f"Scanning {raw_dir} ...")
    index = scan_targeted_results(raw_dir)
    if not index:
        print("No targeted result files found. Run eval_targeted.py first.")
        return
    print(f"  Found {len(index)} result files.")

    data = aggregate_all(raw_dir)
    print_summary(data)

    write_json(data, os.path.join(out_dir, "summary.json"))
    write_csv(build_paper_table(data),    os.path.join(out_dir, "adversarial_summary.csv"))
    write_csv(build_per_spec_table(data), os.path.join(out_dir, "per_spec.csv"))
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Aggregate targeted perturbation results.")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
