"""
Run inference on pre-generated corrupted VSR images.

Loads corrupted JPEGs from data/corrupted/<subdir>/ and runs the specified
inference method, reusing captions, labels, and target boxes from the clean
VSR examples.

RSA is only computed for corruptions where rsa_valid=True (blur, noise).
Rotation corruptions set iou/rsa to None — geometry changes and bounding box
re-projection under rotation is not implemented.

Output:
  results/corrupted/<method>/<display_name>.json

Usage:
  python src/run_corrupted.py --method textual --corruption blur-1
  python src/run_corrupted.py --method visual  --corruption noise-005
  python src/run_corrupted.py --method verification --corruption rot-45

Valid corruptions: blur-1 blur-3 blur-5 noise-001 noise-005 noise-010 rot-15 rot-45 rot-90
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from bqs import compute_bqs
from corruptions import CorruptionSpec, get_all_corruption_specs
from load_vsr import DEV_SUBSET_SIZE, load_vsr_dev_subset
from metrics import compute_rsa, iou as compute_iou
from model_interface import QwenVLLocalInterface
from parse_outputs import (
    extract_json_object, parse_answer, parse_box,
    parse_full_output, parse_reasoning,
)
from utils import format_prompt, format_prompt_multi, load_prompt_template, save_results

_ROOT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N_EXAMPLES    = DEV_SUBSET_SIZE
RSA_THRESHOLD = 0.5
BOX_EPS       = 1e-4


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _get_spec(display_name: str) -> CorruptionSpec:
    for spec in get_all_corruption_specs():
        if spec.display_name == display_name:
            return spec
    valid = [s.display_name for s in get_all_corruption_specs()]
    raise ValueError(f"Unknown corruption {display_name!r}. Valid options: {valid}")


def _load_corrupted_images(examples, spec: CorruptionSpec) -> dict:
    """Return {example_id: PIL Image} mapping from pre-saved corrupted JPEGs."""
    corrupted_root = os.path.join(_ROOT, "data", "corrupted")
    images = {}
    missing = 0
    for ex in examples:
        path = os.path.join(corrupted_root, spec.subdir, f"{ex.example_id}.jpg")
        if os.path.exists(path):
            images[ex.example_id] = Image.open(path).convert("RGB")
        else:
            print(f"  WARNING: {path} not found — using clean image as fallback")
            images[ex.example_id] = ex.image
            missing += 1
    if missing:
        print(f"  {missing}/{len(examples)} images missing. Run generate_corrupted_dataset.py first.")
    return images


def _corruption_meta(spec: CorruptionSpec) -> dict:
    return {
        "corruption_type": spec.corruption_type,
        "severity":        spec.severity,
        "display_name":    spec.display_name,
        "rsa_valid":       spec.rsa_valid,
    }


# ---------------------------------------------------------------------------
# Method 1: Textual CoT
# ---------------------------------------------------------------------------

def _run_textual(examples, images, spec, model) -> list[dict]:
    template = load_prompt_template("textual_cot")
    results = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        prompt = format_prompt(template, ex.caption)

        try:
            raw = model.generate_response(images[ex.example_id], prompt)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            raw = ""

        parsed     = parse_full_output(raw, mode="textual_cot")
        gt_str     = "true" if ex.label else "false"
        parsed_ans = parsed["parsed_answer"]
        correct    = (parsed_ans == gt_str) if parsed_ans is not None else False

        print(f"  ans={parsed_ans}  gt={gt_str}  correct={correct}")

        results.append({
            "example_id":         ex.example_id,
            "caption":            ex.caption,
            "relation":           ex.relation,
            "subj":               ex.subj,
            "obj":                ex.obj,
            "ground_truth":       gt_str,
            "raw_output":         raw,
            "parsed_json":        parsed["parsed_json"],
            "parsed_reasoning":   parsed["parsed_reasoning"],
            "parsed_answer":      parsed_ans,
            "answer_correct":     correct,
            "parsed_box":         parsed["parsed_box"],
            "box_valid":          parsed["box_valid"],
            "box_invalid_reason": parsed["box_invalid_reason"],
            "target_box":         ex.target_box_normalized(),
            "iou":                None,
            "rsa":                None,
            **_corruption_meta(spec),
        })

    return results


# ---------------------------------------------------------------------------
# Method 2: Visual CoT
# ---------------------------------------------------------------------------

def _run_visual(examples, images, spec, model) -> list[dict]:
    template = load_prompt_template("visual_cot")
    results = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        prompt = format_prompt(template, ex.caption)

        try:
            raw = model.generate_response(images[ex.example_id], prompt)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            raw = ""

        parsed     = parse_full_output(raw, mode="visual_cot")
        gt_str     = "true" if ex.label else "false"
        parsed_ans = parsed["parsed_answer"]
        correct    = (parsed_ans == gt_str) if parsed_ans is not None else False
        target_box = ex.target_box_normalized()
        pred_box   = parsed["parsed_box"]

        # RSA only when corruption preserves image geometry
        iou_score = rsa_score = None
        if spec.rsa_valid and parsed["box_valid"] and pred_box and target_box:
            iou_score = compute_iou(pred_box, target_box)
            rsa_score = compute_rsa(pred_box, target_box, RSA_THRESHOLD)

        rsa_str = f"  iou={iou_score:.3f}" if iou_score is not None else ("  RSA=N/A(rotation)" if not spec.rsa_valid else "")
        print(f"  ans={parsed_ans}  correct={correct}  box_valid={parsed['box_valid']}{rsa_str}")

        results.append({
            "example_id":         ex.example_id,
            "caption":            ex.caption,
            "relation":           ex.relation,
            "subj":               ex.subj,
            "obj":                ex.obj,
            "ground_truth":       gt_str,
            "raw_output":         raw,
            "parsed_json":        parsed["parsed_json"],
            "parsed_reasoning":   parsed["parsed_reasoning"],
            "parsed_answer":      parsed_ans,
            "answer_correct":     correct,
            "parsed_box":         pred_box,
            "box_valid":          parsed["box_valid"],
            "box_invalid_reason": parsed["box_invalid_reason"],
            "target_box":         target_box,
            "iou":                iou_score,
            "rsa":                rsa_score,
            **_corruption_meta(spec),
        })

    return results


# ---------------------------------------------------------------------------
# Method 3: Visual CoT + Verification (two-pass)
# ---------------------------------------------------------------------------

def _boxes_differ(a, b) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return any(abs(x - y) > BOX_EPS for x, y in zip(a, b))


def _fmt_box(box) -> str:
    if box is None:
        return "none (no valid box was identified in pass 1)"
    return str([round(v, 4) for v in box])


def _parse_p1(raw: str) -> dict:
    pj = extract_json_object(raw)
    br = parse_box(pj or raw, mode="visual_cot")
    return {
        "pass1_raw_output":         raw,
        "pass1_parsed_json":        pj,
        "pass1_reasoning":          parse_reasoning(pj or raw, key="reasoning"),
        "pass1_box":                br["box"],
        "pass1_box_valid":          br["valid"],
        "pass1_box_invalid_reason": br["invalid_reason"],
        "pass1_answer":             parse_answer(pj or raw),
    }


def _parse_p2(raw: str) -> dict:
    pj = extract_json_object(raw)
    br = parse_box(pj or raw, mode="visual_cot")
    return {
        "pass2_raw_output":              raw,
        "pass2_parsed_json":             pj,
        "pass2_verification_reasoning":  parse_reasoning(pj or raw, key="verification_reasoning"),
        "pass2_box":                     br["box"],
        "pass2_box_valid":               br["valid"],
        "pass2_box_invalid_reason":      br["invalid_reason"],
        "pass2_answer":                  parse_answer(pj or raw),
    }


def _run_verification(examples, images, spec, model) -> list[dict]:
    p1_tmpl = load_prompt_template("visual_cot_pass1")
    p2_tmpl = load_prompt_template("visual_cot_verification_pass2")
    results = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        img        = images[ex.example_id]
        gt_str     = "true" if ex.label else "false"
        target_box = ex.target_box_normalized()

        # Pass 1
        try:
            p1_raw = model.generate_response(img, format_prompt(p1_tmpl, ex.caption))
        except Exception as exc:
            print(f"  [P1] ERROR: {exc}"); p1_raw = ""

        p1          = _parse_p1(p1_raw)
        p1_correct  = (p1["pass1_answer"] == gt_str) if p1["pass1_answer"] else False

        p1_iou = p1_rsa = None
        if spec.rsa_valid and p1["pass1_box_valid"] and p1["pass1_box"] and target_box:
            p1_iou = compute_iou(p1["pass1_box"], target_box)
            p1_rsa = compute_rsa(p1["pass1_box"], target_box, RSA_THRESHOLD)

        print(f"  [P1] ans={p1['pass1_answer']}  correct={p1_correct}"
              + (f"  iou={p1_iou:.3f}" if p1_iou is not None else ""))

        # Pass 2
        p2_prompt = format_prompt_multi(p2_tmpl, caption=ex.caption,
                                        initial_box=_fmt_box(p1["pass1_box"]))
        try:
            p2_raw = model.generate_response(img, p2_prompt)
        except Exception as exc:
            print(f"  [P2] ERROR: {exc}"); p2_raw = ""

        p2         = _parse_p2(p2_raw)
        p2_correct = (p2["pass2_answer"] == gt_str) if p2["pass2_answer"] else False

        p2_iou = p2_rsa = None
        if spec.rsa_valid and p2["pass2_box_valid"] and p2["pass2_box"] and target_box:
            p2_iou = compute_iou(p2["pass2_box"], target_box)
            p2_rsa = compute_rsa(p2["pass2_box"], target_box, RSA_THRESHOLD)

        box_revised    = _boxes_differ(p1["pass1_box"], p2["pass2_box"])
        answer_flipped = (p1["pass1_answer"] is not None and p2["pass2_answer"] is not None
                          and p1["pass1_answer"] != p2["pass2_answer"])
        recovered      = (not p1_correct) and p2_correct

        print(f"  [P2] ans={p2['pass2_answer']}  correct={p2_correct}"
              f"  revised={box_revised}  flipped={answer_flipped}")

        results.append({
            "example_id":           ex.example_id,
            "caption":              ex.caption,
            "relation":             ex.relation,
            "subj":                 ex.subj,
            "obj":                  ex.obj,
            "ground_truth_answer":  gt_str,
            "target_box":           target_box,
            **p1,
            "pass1_answer_correct": p1_correct,
            "pass1_iou":            p1_iou,
            "pass1_rsa":            p1_rsa,
            **p2,
            "pass2_answer_correct": p2_correct,
            "pass2_iou":            p2_iou,
            "pass2_rsa":            p2_rsa,
            "box_revised":          box_revised,
            "answer_flipped":       answer_flipped,
            "recovered":            recovered,
            **_corruption_meta(spec),
        })

    return results


# ---------------------------------------------------------------------------
# Method 4: Multi-Stage Correction (corrupted images)
# ---------------------------------------------------------------------------

_BQS_THRESHOLD = 0.6


def _run_multistage(examples, images, spec, model) -> list[dict]:
    """
    Multi-stage correction on corrupted images.

    Mirrors run_multistage_correction._process_example but sources the image
    from the pre-saved corrupted JPEG rather than ex.image.

    RSA is skipped for rotation (spec.rsa_valid=False) just like other methods.
    """
    p1_tmpl    = load_prompt_template("multi_stage_pass1")
    verif_tmpl = load_prompt_template("multi_stage_verification")
    reask_tmpl = load_prompt_template("multi_stage_reask_answer")
    results    = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        img        = images[ex.example_id]   # corrupted image
        gt         = "true" if ex.label else "false"
        target_box = ex.target_box_normalized()

        # ---- Stage 1: initial grounding ----
        try:
            p1_raw = model.generate_response(img, format_prompt(p1_tmpl, ex.caption))
        except Exception as exc:
            print(f"  [S1] ERROR: {exc}"); p1_raw = ""

        pj1  = extract_json_object(p1_raw)
        br1  = parse_box(pj1 or p1_raw, mode="visual_cot")
        s1   = {
            "initial_raw_output":         p1_raw,
            "initial_parsed_json":        pj1,
            "initial_reasoning":          parse_reasoning(pj1 or p1_raw, key="reasoning"),
            "initial_box":                br1["box"],
            "initial_box_valid":          br1["valid"],
            "initial_box_invalid_reason": br1["invalid_reason"],
            "initial_answer":             parse_answer(pj1 or p1_raw),
        }
        init_correct = (s1["initial_answer"] == gt) if s1["initial_answer"] else False

        init_iou = init_rsa = None
        if spec.rsa_valid and br1["valid"] and br1["box"] and target_box:
            init_iou = compute_iou(br1["box"], target_box)
            init_rsa = compute_rsa(br1["box"], target_box, RSA_THRESHOLD)

        print(f"  [S1] ans={s1['initial_answer']}  correct={init_correct}"
              f"  box_valid={br1['valid']}"
              + (f"  iou={init_iou:.3f}" if init_iou is not None else ""))

        # ---- Stage 2: BQS (provisional) ----
        bqs_prov = compute_bqs(
            box_valid=br1["valid"], box=br1["box"], target_box=target_box,
            reasoning=s1["initial_reasoning"], subj=ex.subj, obj=ex.obj,
            stability_score=1.0,
        )
        triggered = not br1["valid"]

        # ---- Stages 4 & 5: correction ----
        if triggered:
            v_prompt = format_prompt_multi(
                verif_tmpl, caption=ex.caption,
                initial_box=_fmt_box(br1["box"]),
            )
            try:
                v_raw = model.generate_response(img, v_prompt)
            except Exception as exc:
                print(f"  [S4] ERROR: {exc}"); v_raw = ""

            pj4 = extract_json_object(v_raw)
            br4 = parse_box(pj4 or v_raw, mode="visual_cot")
            s4  = {
                "verification_raw_output":    v_raw,
                "verification_parsed_json":   pj4,
                "verification_reasoning":     parse_reasoning(pj4 or v_raw, key="verification_reasoning"),
                "revised_box":                br4["box"],
                "revised_box_valid":          br4["valid"],
                "revised_box_invalid_reason": br4["invalid_reason"],
                "revised_answer":             parse_answer(pj4 or v_raw),
            }
            rev_iou = rev_rsa = None
            if spec.rsa_valid and br4["valid"] and br4["box"] and target_box:
                rev_iou = compute_iou(br4["box"], target_box)
                rev_rsa = compute_rsa(br4["box"], target_box, RSA_THRESHOLD)

            try:
                c_raw = model.generate_response(
                    img,
                    format_prompt_multi(reask_tmpl, caption=ex.caption,
                                        revised_box=_fmt_box(br4["box"])),
                )
            except Exception as exc:
                print(f"  [S5] ERROR: {exc}"); c_raw = ""

            pj5          = extract_json_object(c_raw)
            final_answer = parse_answer(pj5 or c_raw)
            stability    = 1.0 if final_answer == s1["initial_answer"] else 0.0
            bqs_final    = compute_bqs(
                box_valid=br1["valid"], box=br1["box"], target_box=target_box,
                reasoning=s1["initial_reasoning"], subj=ex.subj, obj=ex.obj,
                stability_score=stability,
            )
            reask_result = {
                "reask_raw_output":  c_raw,
                "reask_parsed_json": pj5,
                "reask_reasoning":   parse_reasoning(pj5 or c_raw, key="answer_reasoning"),
            }
        else:
            s4 = {k: None for k in (
                "verification_raw_output", "verification_parsed_json",
                "verification_reasoning", "revised_box", "revised_box_valid",
                "revised_box_invalid_reason", "revised_answer",
            )}
            reask_result = {"reask_raw_output": None, "reask_parsed_json": None, "reask_reasoning": None}
            rev_iou = rev_rsa = None
            final_answer  = s1["initial_answer"]
            stability     = 1.0
            bqs_final     = bqs_prov

        final_correct = (final_answer == gt) if final_answer else False
        rev_box       = s4["revised_box"]
        box_revised   = _boxes_differ(s1["initial_box"], rev_box) if triggered else False
        ans_flipped   = (s1["initial_answer"] is not None and final_answer is not None
                         and s1["initial_answer"] != final_answer)

        print(f"  [FN] final={final_answer}  correct={final_correct}"
              f"  revised={box_revised}  flipped={ans_flipped}")

        results.append({
            "example_id":          ex.example_id,
            "caption":             ex.caption,
            "relation":            ex.relation,
            "subj":                ex.subj,
            "obj":                 ex.obj,
            "ground_truth_answer": gt,
            "target_box":          target_box,
            **s1,
            "initial_answer_correct": init_correct,
            "initial_iou":            init_iou,
            "initial_rsa":            init_rsa,
            "bqs":                    bqs_final["bqs"],
            "format_score":           bqs_final["format_score"],
            "overlap_score":          bqs_final["overlap_score"],
            "mention_score":          bqs_final["mention_score"],
            "stability_score":        bqs_final["stability_score"],
            "correction_triggered":   triggered,
            **s4,
            "revised_iou":            rev_iou,
            "revised_rsa":            rev_rsa,
            **reask_result,
            "final_answer":           final_answer,
            "final_answer_correct":   final_correct,
            "box_revised":            box_revised,
            "answer_flipped":         ans_flipped,
            "recovered":              (not init_correct) and final_correct,
            **_corruption_meta(spec),
        })

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], method: str, spec: CorruptionSpec) -> None:
    n = len(results)
    if method == "textual":
        c = sum(1 for r in results if r.get("answer_correct"))
        print(f"\n  [{spec.display_name}] Textual CoT  FAA={c/n:.3f} ({c}/{n})")
    elif method == "visual":
        c = sum(1 for r in results if r.get("answer_correct"))
        rsa_vals = [r["rsa"] for r in results if r.get("rsa") is not None]
        rsa = f"{sum(rsa_vals)/len(rsa_vals):.3f}" if rsa_vals else "N/A"
        print(f"\n  [{spec.display_name}] Visual CoT   FAA={c/n:.3f} ({c}/{n})  RSA={rsa}")
    else:
        c = sum(1 for r in results if r.get("pass2_answer_correct"))
        rsa_vals = [r["pass2_rsa"] for r in results if r.get("pass2_rsa") is not None]
        rsa = f"{sum(rsa_vals)/len(rsa_vals):.3f}" if rsa_vals else "N/A"
        print(f"\n  [{spec.display_name}] Verification FAA={c/n:.3f} ({c}/{n})  RSA={rsa}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on corrupted VSR images.")
    parser.add_argument("--method", required=True,
                        choices=["textual", "visual", "verification", "multistage"])
    parser.add_argument("--corruption", required=True,
                        help="e.g. blur-1, noise-005, rot-45")
    args = parser.parse_args()

    spec = _get_spec(args.corruption)
    print(f"Method: {args.method}  Corruption: {spec.display_name}  "
          f"RSA valid: {spec.rsa_valid}")

    print("\nLoading VSR examples...")
    examples = load_vsr_dev_subset(n=N_EXAMPLES)
    print(f"  {len(examples)} examples ready.")

    print("\nLoading corrupted images from disk...")
    images = _load_corrupted_images(examples, spec)

    print("\nLoading model...")
    model = QwenVLLocalInterface()

    dispatch = {
        "textual":      _run_textual,
        "visual":       _run_visual,
        "verification": _run_verification,
        "multistage":   _run_multistage,
    }
    results = dispatch[args.method](examples, images, spec, model)

    out_dir  = os.path.join(_ROOT, "results", "corrupted", args.method)
    out_path = os.path.join(out_dir, f"{args.corruption}.json")
    save_results(results, out_path)
    _print_summary(results, args.method, spec)


if __name__ == "__main__":
    main()
