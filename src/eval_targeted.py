"""
Evaluate all reasoning methods on targeted perturbation images.

Runs one (model, method, perturbation_spec) cell at a time, saving results to
results/targeted_raw/{model}_{method}_{display_name}.json

This mirrors eval_runner.py but uses perturbed images from data/targeted/ instead
of clean or corruption-based images.

RSA is valid for all targeted perturbations (geometry is preserved).

Usage:
  python src/eval_targeted.py \\
      --model qwen \\
      --method visual \\
      --perturbation occlusion-obj1-low \\
      --targeted-dir data/targeted \\
      --subset data/subsets/vsr_n200_seq.json

  # All 4 methods × 15 specs (60 jobs total):
  for method in textual visual verification multistage; do
    for spec in occlusion-obj1-low occlusion-obj1-medium occlusion-obj1-high \\
                occlusion-obj2-low occlusion-obj2-medium occlusion-obj2-high \\
                mask-union-low mask-union-medium mask-union-high \\
                distractor-nearobj1-low distractor-nearobj1-medium distractor-nearobj1-high \\
                distractor-nearobj2-low distractor-nearobj2-medium distractor-nearobj2-high; do
      sbatch scripts/run_eval_targeted.sh $method $spec
    done
  done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from adversarial_perturbations import PerturbationSpec, get_all_targeted_perturbation_specs
from bqs import compute_bqs
from eval_config import EvalConfig, load_subset_indices
from load_vsr import ANNOTATION_FILE, load_vsr_by_indices, load_vsr_dev_subset
from metrics import compute_rsa, iou as compute_iou
from model_interface import QwenVLLocalInterface
from parse_outputs import (
    extract_json_object, parse_answer, parse_box, parse_reasoning,
    parse_full_output,
)
from utils import format_prompt, format_prompt_multi, load_prompt_template, save_results

_ROOT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RSA_THRESHOLD = 0.5
BOX_EPS       = 1e-4

# Index of all perturbation specs by display_name for fast lookup
_SPEC_INDEX: dict[str, PerturbationSpec] = {
    s.display_name: s for s in get_all_targeted_perturbation_specs()
}


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_targeted_metadata(targeted_dir: str) -> dict:
    """Load the metadata.json produced by generate_targeted_perturbations.py."""
    path = (targeted_dir if os.path.isabs(targeted_dir)
            else os.path.join(_ROOT, targeted_dir))
    meta_path = os.path.join(path, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"metadata.json not found in {path}. "
            f"Run generate_targeted_perturbations.py first."
        )
    with open(meta_path) as f:
        return json.load(f)


def _build_image_index(meta: dict, spec: PerturbationSpec) -> dict[str, str]:
    """
    Return {example_id: absolute_image_path} for examples that have a valid
    perturbed image for `spec`.
    """
    index = {}
    for ex_rec in meta.get("examples", []):
        if not ex_rec.get("has_boxes"):
            continue
        perts = ex_rec.get("perturbations", {})
        p = perts.get(spec.display_name, {})
        if p.get("image_path") and not p.get("skip_reason"):
            rel = p["image_path"]
            abs_path = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
            if os.path.exists(abs_path):
                index[ex_rec["example_id"]] = abs_path
    return index


# ---------------------------------------------------------------------------
# Common helpers (mirrors eval_runner.py)
# ---------------------------------------------------------------------------

def _ptmeta(model_name: str, method: str, spec: PerturbationSpec) -> dict:
    return {
        "model":            model_name,
        "method":           method,
        "perturbation_type": spec.perturbation_type,
        "target":           spec.target,
        "severity":         spec.severity,
        "display_name":     spec.display_name,
        "rsa_valid":        spec.rsa_valid,
    }


def _spatial(box_valid, box, target_box, rsa_valid):
    """Compute (iou, rsa); None when RSA is invalid or boxes absent."""
    if not rsa_valid or not box_valid or not box or not target_box:
        return None, None
    iou_val = compute_iou(box, target_box)
    rsa_val = compute_rsa(box, target_box, RSA_THRESHOLD)
    return iou_val, rsa_val


def _boxes_differ(a, b) -> bool:
    if a is None and b is None: return False
    if a is None or b is None:  return True
    return any(abs(x - y) > BOX_EPS for x, y in zip(a, b))


def _fmt_box(box) -> str:
    if box is None:
        return "none (no valid box was identified)"
    return str([round(v, 4) for v in box])


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def _run_textual(examples, images, spec, model, model_name, method) -> list[dict]:
    template = load_prompt_template("textual_cot")
    meta     = _ptmeta(model_name, method, spec)
    results  = []

    for i, ex in enumerate(examples):
        if ex.example_id not in images:
            continue
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        try:
            raw = model.generate_response(images[ex.example_id],
                                          format_prompt(template, ex.caption))
        except Exception as exc:
            print(f"  ERROR: {exc}"); raw = ""

        parsed = parse_full_output(raw, mode="textual_cot")
        gt     = "true" if ex.label else "false"
        ans    = parsed["parsed_answer"]
        correct = (ans == gt) if ans is not None else False

        print(f"  ans={ans}  gt={gt}  correct={correct}")
        results.append({
            "example_id":         ex.example_id,
            "caption":            ex.caption,
            "relation":           ex.relation,
            "subj":               ex.subj,
            "obj":                ex.obj,
            "ground_truth":       gt,
            "raw_output":         raw,
            "parsed_json":        parsed["parsed_json"],
            "parsed_reasoning":   parsed["parsed_reasoning"],
            "parsed_answer":      ans,
            "answer_correct":     correct,
            "target_box":         ex.target_box_normalized(),
            **meta,
        })

    return results


def _run_visual(examples, images, spec, model, model_name, method) -> list[dict]:
    template = load_prompt_template("visual_cot")
    meta     = _ptmeta(model_name, method, spec)
    results  = []

    for i, ex in enumerate(examples):
        if ex.example_id not in images:
            continue
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        try:
            raw = model.generate_response(images[ex.example_id],
                                          format_prompt(template, ex.caption))
        except Exception as exc:
            print(f"  ERROR: {exc}"); raw = ""

        parsed     = parse_full_output(raw, mode="visual_cot")
        gt         = "true" if ex.label else "false"
        ans        = parsed["parsed_answer"]
        correct    = (ans == gt) if ans is not None else False
        target_box = ex.target_box_normalized()
        iou_val, rsa_val = _spatial(
            parsed["box_valid"], parsed["parsed_box"], target_box, spec.rsa_valid
        )

        print(f"  ans={ans}  correct={correct}  box_valid={parsed['box_valid']}"
              + (f"  iou={iou_val:.3f}" if iou_val is not None else ""))
        results.append({
            "example_id":         ex.example_id,
            "caption":            ex.caption,
            "relation":           ex.relation,
            "subj":               ex.subj,
            "obj":                ex.obj,
            "ground_truth":       gt,
            "raw_output":         raw,
            "parsed_json":        parsed["parsed_json"],
            "parsed_reasoning":   parsed["parsed_reasoning"],
            "parsed_answer":      ans,
            "answer_correct":     correct,
            "parsed_box":         parsed["parsed_box"],
            "box_valid":          parsed["box_valid"],
            "box_invalid_reason": parsed["box_invalid_reason"],
            "target_box":         target_box,
            "iou":                iou_val,
            "rsa":                rsa_val,
            **meta,
        })

    return results


def _run_verification(examples, images, spec, model, model_name, method) -> list[dict]:
    p1_tmpl = load_prompt_template("visual_cot_pass1")
    p2_tmpl = load_prompt_template("visual_cot_verification_pass2")
    meta    = _ptmeta(model_name, method, spec)
    results = []

    for i, ex in enumerate(examples):
        if ex.example_id not in images:
            continue
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        img        = images[ex.example_id]
        gt         = "true" if ex.label else "false"
        target_box = ex.target_box_normalized()

        try:
            p1_raw = model.generate_response(img, format_prompt(p1_tmpl, ex.caption))
        except Exception as exc:
            print(f"  [P1] ERROR: {exc}"); p1_raw = ""

        pj1        = extract_json_object(p1_raw)
        br1        = parse_box(pj1 or p1_raw, mode="visual_cot")
        p1_ans     = parse_answer(pj1 or p1_raw)
        p1_correct = (p1_ans == gt) if p1_ans else False
        p1_iou, p1_rsa = _spatial(br1["valid"], br1["box"], target_box, spec.rsa_valid)

        print(f"  [P1] ans={p1_ans}  correct={p1_correct}"
              + (f"  iou={p1_iou:.3f}" if p1_iou is not None else ""))

        try:
            p2_raw = model.generate_response(
                img,
                format_prompt_multi(p2_tmpl, caption=ex.caption,
                                    initial_box=_fmt_box(br1["box"])),
            )
        except Exception as exc:
            print(f"  [P2] ERROR: {exc}"); p2_raw = ""

        pj2        = extract_json_object(p2_raw)
        br2        = parse_box(pj2 or p2_raw, mode="visual_cot")
        p2_ans     = parse_answer(pj2 or p2_raw)
        p2_correct = (p2_ans == gt) if p2_ans else False
        p2_iou, p2_rsa = _spatial(br2["valid"], br2["box"], target_box, spec.rsa_valid)

        box_revised    = _boxes_differ(br1["box"], br2["box"])
        answer_flipped = (p1_ans is not None and p2_ans is not None and p1_ans != p2_ans)

        print(f"  [P2] ans={p2_ans}  correct={p2_correct}"
              f"  revised={box_revised}  flipped={answer_flipped}")

        results.append({
            "example_id":                   ex.example_id,
            "caption":                      ex.caption,
            "relation":                     ex.relation,
            "subj":                         ex.subj,
            "obj":                          ex.obj,
            "ground_truth_answer":          gt,
            "target_box":                   target_box,
            "pass1_raw_output":             p1_raw,
            "pass1_parsed_json":            pj1,
            "pass1_reasoning":              parse_reasoning(pj1 or p1_raw, key="reasoning"),
            "pass1_box":                    br1["box"],
            "pass1_box_valid":              br1["valid"],
            "pass1_box_invalid_reason":     br1["invalid_reason"],
            "pass1_answer":                 p1_ans,
            "pass1_answer_correct":         p1_correct,
            "pass1_iou":                    p1_iou,
            "pass1_rsa":                    p1_rsa,
            "pass2_raw_output":             p2_raw,
            "pass2_parsed_json":            pj2,
            "pass2_verification_reasoning": parse_reasoning(pj2 or p2_raw,
                                                            key="verification_reasoning"),
            "pass2_box":                    br2["box"],
            "pass2_box_valid":              br2["valid"],
            "pass2_box_invalid_reason":     br2["invalid_reason"],
            "pass2_answer":                 p2_ans,
            "pass2_answer_correct":         p2_correct,
            "pass2_iou":                    p2_iou,
            "pass2_rsa":                    p2_rsa,
            "box_revised":                  box_revised,
            "answer_flipped":               answer_flipped,
            "recovered":                    (not p1_correct) and p2_correct,
            **meta,
        })

    return results


def _run_multistage(examples, images, spec, model, model_name, method) -> list[dict]:
    p1_tmpl    = load_prompt_template("multi_stage_pass1")
    verif_tmpl = load_prompt_template("multi_stage_verification")
    reask_tmpl = load_prompt_template("multi_stage_reask_answer")
    meta       = _ptmeta(model_name, method, spec)
    BQS_THRESHOLD = 0.6
    results    = []

    for i, ex in enumerate(examples):
        if ex.example_id not in images:
            continue
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        img        = images[ex.example_id]
        gt         = "true" if ex.label else "false"
        target_box = ex.target_box_normalized()

        try:
            p1_raw = model.generate_response(img, format_prompt(p1_tmpl, ex.caption))
        except Exception as exc:
            print(f"  [S1] ERROR: {exc}"); p1_raw = ""

        pj1 = extract_json_object(p1_raw)
        br1 = parse_box(pj1 or p1_raw, mode="visual_cot")
        s1  = {
            "initial_raw_output":         p1_raw,
            "initial_parsed_json":        pj1,
            "initial_reasoning":          parse_reasoning(pj1 or p1_raw, key="reasoning"),
            "initial_box":                br1["box"],
            "initial_box_valid":          br1["valid"],
            "initial_box_invalid_reason": br1["invalid_reason"],
            "initial_answer":             parse_answer(pj1 or p1_raw),
        }
        init_correct = (s1["initial_answer"] == gt) if s1["initial_answer"] else False
        init_iou, init_rsa = _spatial(br1["valid"], br1["box"], target_box, spec.rsa_valid)

        print(f"  [S1] ans={s1['initial_answer']}  correct={init_correct}"
              f"  box_valid={br1['valid']}"
              + (f"  iou={init_iou:.3f}" if init_iou is not None else ""))

        bqs_prov  = compute_bqs(
            box_valid=br1["valid"], box=br1["box"], target_box=target_box,
            reasoning=s1["initial_reasoning"], subj=ex.subj, obj=ex.obj,
            stability_score=1.0,
        )
        triggered = not br1["valid"]

        if triggered:
            try:
                v_raw = model.generate_response(
                    img,
                    format_prompt_multi(verif_tmpl, caption=ex.caption,
                                        initial_box=_fmt_box(br1["box"])),
                )
            except Exception as exc:
                print(f"  [S4] ERROR: {exc}"); v_raw = ""

            pj4 = extract_json_object(v_raw)
            br4 = parse_box(pj4 or v_raw, mode="visual_cot")
            s4  = {
                "verification_raw_output":    v_raw,
                "verification_parsed_json":   pj4,
                "verification_reasoning":     parse_reasoning(pj4 or v_raw,
                                                              key="verification_reasoning"),
                "revised_box":                br4["box"],
                "revised_box_valid":          br4["valid"],
                "revised_box_invalid_reason": br4["invalid_reason"],
                "revised_answer":             parse_answer(pj4 or v_raw),
            }
            rev_iou, rev_rsa = _spatial(br4["valid"], br4["box"], target_box, spec.rsa_valid)

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
            reask_fields = {
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
            reask_fields = {"reask_raw_output": None, "reask_parsed_json": None,
                            "reask_reasoning": None}
            rev_iou = rev_rsa = None
            final_answer = s1["initial_answer"]
            stability    = 1.0
            bqs_final    = bqs_prov

        final_correct  = (final_answer == gt) if final_answer else False
        box_revised    = _boxes_differ(s1["initial_box"], s4["revised_box"]) if triggered else False
        answer_flipped = (s1["initial_answer"] is not None and final_answer is not None
                          and s1["initial_answer"] != final_answer)

        print(f"  [FN] final={final_answer}  correct={final_correct}"
              f"  revised={box_revised}  flipped={answer_flipped}")

        results.append({
            "example_id":             ex.example_id,
            "caption":                ex.caption,
            "relation":               ex.relation,
            "subj":                   ex.subj,
            "obj":                    ex.obj,
            "ground_truth_answer":    gt,
            "target_box":             target_box,
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
            **reask_fields,
            "final_answer":           final_answer,
            "final_answer_correct":   final_correct,
            "box_revised":            box_revised,
            "answer_flipped":         answer_flipped,
            "recovered":              (not init_correct) and final_correct,
            **meta,
        })

    return results


_RUNNERS = {
    "textual":      _run_textual,
    "visual":       _run_visual,
    "verification": _run_verification,
    "multistage":   _run_multistage,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval_targeted(
    model_name: str,
    method: str,
    perturbation_display_name: str,
    targeted_dir: str,
    subset_file: Optional[str],
    n_fallback: int,
    out_dir: str,
    max_new_tokens: int,
) -> list[dict]:
    """
    Run one (model, method, perturbation_spec) evaluation cell.
    Saves results to {out_dir}/{model_name}_{method}_{display_name}.json.
    """
    if perturbation_display_name not in _SPEC_INDEX:
        raise ValueError(
            f"Unknown perturbation {perturbation_display_name!r}. "
            f"Valid: {sorted(_SPEC_INDEX)}"
        )
    spec = _SPEC_INDEX[perturbation_display_name]

    print(f"\n{'='*60}")
    print(f"  model={model_name}  method={method}  perturbation={spec.display_name}")
    print(f"{'='*60}")

    # Load metadata and image paths
    print("\nLoading targeted metadata...")
    meta = load_targeted_metadata(targeted_dir)
    img_index = _build_image_index(meta, spec)
    print(f"  {len(img_index)} perturbed images available for {spec.display_name}.")

    if not img_index:
        print("  No images available — skipping.")
        return []

    # Load VSR examples (need captions, labels, etc.)
    print("\nLoading VSR examples...")
    if subset_file:
        path = subset_file if os.path.isabs(subset_file) else os.path.join(_ROOT, subset_file)
        indices = load_subset_indices(path)
        examples = load_vsr_by_indices(indices, annotation_file=ANNOTATION_FILE)
    else:
        examples = load_vsr_dev_subset(n=n_fallback, annotation_file=ANNOTATION_FILE)

    # Filter to examples that have perturbed images
    examples = [ex for ex in examples if ex.example_id in img_index]
    print(f"  {len(examples)} examples with perturbed images.")

    # Load perturbed images into memory
    print("\nLoading perturbed images...")
    images: dict[str, Image.Image] = {}
    for ex in examples:
        try:
            images[ex.example_id] = Image.open(img_index[ex.example_id]).convert("RGB")
        except Exception as exc:
            print(f"  WARNING: could not load {img_index[ex.example_id]}: {exc}")

    # Load model
    print("\nLoading model...")
    if model_name == "qwen":
        model = QwenVLLocalInterface(max_new_tokens=max_new_tokens)
    else:
        raise ValueError(f"Unknown model {model_name!r}")

    # Run
    runner  = _RUNNERS[method]
    results = runner(examples, images, spec, model, model_name, method)

    # Save
    out_path = os.path.join(
        out_dir if os.path.isabs(out_dir) else os.path.join(_ROOT, out_dir),
        f"{model_name}_{method}_{spec.display_name}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_results(results, out_path)

    n = len(results)
    if method in ("textual", "visual"):
        correct = sum(1 for r in results if r.get("answer_correct"))
    elif method == "verification":
        correct = sum(1 for r in results if r.get("pass2_answer_correct"))
    else:
        correct = sum(1 for r in results if r.get("final_answer_correct"))
    if n:
        print(f"\n  FAA = {correct/n:.3f} ({correct}/{n})")

    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run one (model, method, perturbation) targeted evaluation cell.")
    p.add_argument("--model",        default="qwen", choices=["qwen"])
    p.add_argument("--method",       required=True,
                   choices=["textual", "visual", "verification", "multistage"])
    p.add_argument("--perturbation", required=True,
                   help="Perturbation display_name, e.g. occlusion-obj1-low")
    p.add_argument("--targeted-dir", default="data/targeted",
                   help="Directory with metadata.json and perturbed images")
    p.add_argument("--subset",       default="data/subsets/vsr_n200_seq.json")
    p.add_argument("--n",            type=int, default=200)
    p.add_argument("--out-dir",      default="results/targeted_raw")
    p.add_argument("--max-tokens",   type=int, default=512)
    args = p.parse_args()

    run_eval_targeted(
        model_name                = args.model,
        method                    = args.method,
        perturbation_display_name = args.perturbation,
        targeted_dir              = args.targeted_dir,
        subset_file               = args.subset,
        n_fallback                = args.n,
        out_dir                   = args.out_dir,
        max_new_tokens            = args.max_tokens,
    )


if __name__ == "__main__":
    main()
