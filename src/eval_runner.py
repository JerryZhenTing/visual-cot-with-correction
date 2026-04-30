"""
Unified evaluation runner for the matched VSR pipeline.

This is the single entry point for running any (model, method, condition)
combination on a fixed matched subset.  All comparisons in the paper should
use results produced by this script so that every cell in the table comes
from the same examples.

Workflow:
    # 1. Create a matched subset once:
    python src/eval_config.py --n 200 --seed 42 --out data/subsets/vsr_n200_s42.json

    # 2. Run one cell:
    python src/eval_runner.py \\
        --model qwen --method visual --condition blur-1 \\
        --subset data/subsets/vsr_n200_s42.json

    # 3. Run all cells (loop in a shell or SLURM job array):
    for method in textual visual verification multistage; do
      for condition in clean blur-1 blur-3 blur-5 noise-001 noise-005 noise-010 rot-15 rot-45 rot-90; do
        python src/eval_runner.py --model qwen --method $method --condition $condition \\
            --subset data/subsets/vsr_n200_s42.json
      done
    done

    # 4. Aggregate:
    python src/aggregate_results.py

Output per run: results/raw/{model}_{method}_{condition}.json
Each result dict includes: example_id, model, method, condition, corruption_type,
severity, rsa_valid, ... plus all method-specific fields.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from bqs import compute_bqs
from corruptions import CorruptionSpec, get_all_corruption_specs
from eval_config import EvalConfig, load_subset_indices
from load_vsr import load_vsr_by_indices, load_vsr_dev_subset, ANNOTATION_FILE
from metrics import compute_rsa, iou as compute_iou
from model_interface import QwenVLLocalInterface
from parse_outputs import (
    extract_json_object, parse_answer, parse_box, parse_reasoning,
    parse_full_output,
)
from utils import format_prompt, format_prompt_multi, load_prompt_template, save_results

_ROOT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RSA_THRESHOLD = 0.5
BQS_THRESHOLD = 0.6
BOX_EPS       = 1e-4

# Corruption specs indexed by display_name for fast lookup
_SPEC_MAP: dict[str, CorruptionSpec] = {
    s.display_name: s for s in get_all_corruption_specs()
}

# Synthetic spec for the "clean" condition
_CLEAN_SPEC = CorruptionSpec(
    corruption_type="clean",
    severity=0.0,
    display_name="clean",
    rsa_valid=True,
    subdir="",
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_examples(config: EvalConfig):
    """Load VSR examples — indexed if subset_file provided, else first-N fallback."""
    if config.subset_file:
        path = (config.subset_file if os.path.isabs(config.subset_file)
                else os.path.join(_ROOT, config.subset_file))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Subset file not found: {path}\n"
                f"Create it with: python src/eval_config.py --n {config.n_examples} "
                f"--seed {config.subset_seed} --out {config.subset_file}"
            )
        indices = load_subset_indices(path)
        return load_vsr_by_indices(indices, annotation_file=ANNOTATION_FILE)
    else:
        print(f"[WARN] No subset_file specified — loading first {config.n_examples} examples.")
        return load_vsr_dev_subset(n=config.n_examples)


def _load_images(examples, spec: CorruptionSpec) -> dict:
    """
    Return {example_id: PIL Image} mapping.
    For 'clean', images come from the VSRExample objects directly.
    For corrupted conditions, images are loaded from data/corrupted/<subdir>/.
    """
    if spec.corruption_type == "clean":
        return {ex.example_id: ex.image for ex in examples}

    corrupted_root = os.path.join(_ROOT, "data", "corrupted")
    images, missing = {}, 0
    for ex in examples:
        path = os.path.join(corrupted_root, spec.subdir, f"{ex.example_id}.jpg")
        if os.path.exists(path):
            images[ex.example_id] = Image.open(path).convert("RGB")
        else:
            print(f"  WARNING: {path} not found — using clean image as fallback")
            images[ex.example_id] = ex.image
            missing += 1
    if missing:
        print(f"  {missing}/{len(examples)} images missing. "
              f"Run: python src/generate_corrupted_dataset.py")
    return images


def _load_model(config: EvalConfig) -> QwenVLLocalInterface:
    """Instantiate the model requested in the config."""
    if config.model_name == "qwen":
        return QwenVLLocalInterface(max_new_tokens=config.max_new_tokens)
    raise ValueError(
        f"Unknown model {config.model_name!r}. "
        f"Currently supported: 'qwen'. LLaVA support is planned."
    )


def _meta(config: EvalConfig, spec: CorruptionSpec) -> dict:
    """Common metadata fields added to every result dict."""
    return {
        "model":            config.model_name,
        "method":           config.method,
        "condition":        config.condition,
        "corruption_type":  spec.corruption_type,
        "severity":         spec.severity,
        "rsa_valid":        spec.rsa_valid,
    }


# ---------------------------------------------------------------------------
# Spatial helper
# ---------------------------------------------------------------------------

def _spatial(box_valid, box, target_box, rsa_valid):
    """Return (iou, rsa) — both None when RSA is invalid (rotation) or boxes absent."""
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

def _run_textual(examples, images, spec, model, config) -> list[dict]:
    template = load_prompt_template("textual_cot")
    meta     = _meta(config, spec)
    results  = []

    for i, ex in enumerate(examples):
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


def _run_visual(examples, images, spec, model, config) -> list[dict]:
    template = load_prompt_template("visual_cot")
    meta     = _meta(config, spec)
    results  = []

    for i, ex in enumerate(examples):
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


def _run_verification(examples, images, spec, model, config) -> list[dict]:
    p1_tmpl = load_prompt_template("visual_cot_pass1")
    p2_tmpl = load_prompt_template("visual_cot_verification_pass2")
    meta    = _meta(config, spec)
    results = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        img        = images[ex.example_id]
        gt         = "true" if ex.label else "false"
        target_box = ex.target_box_normalized()

        # Pass 1
        try:
            p1_raw = model.generate_response(img, format_prompt(p1_tmpl, ex.caption))
        except Exception as exc:
            print(f"  [P1] ERROR: {exc}"); p1_raw = ""

        pj1       = extract_json_object(p1_raw)
        br1       = parse_box(pj1 or p1_raw, mode="visual_cot")
        p1_ans    = parse_answer(pj1 or p1_raw)
        p1_correct = (p1_ans == gt) if p1_ans else False
        p1_iou, p1_rsa = _spatial(br1["valid"], br1["box"], target_box, spec.rsa_valid)

        print(f"  [P1] ans={p1_ans}  correct={p1_correct}"
              + (f"  iou={p1_iou:.3f}" if p1_iou is not None else ""))

        # Pass 2
        p2_prompt = format_prompt_multi(
            p2_tmpl, caption=ex.caption, initial_box=_fmt_box(br1["box"])
        )
        try:
            p2_raw = model.generate_response(img, p2_prompt)
        except Exception as exc:
            print(f"  [P2] ERROR: {exc}"); p2_raw = ""

        pj2       = extract_json_object(p2_raw)
        br2       = parse_box(pj2 or p2_raw, mode="visual_cot")
        p2_ans    = parse_answer(pj2 or p2_raw)
        p2_correct = (p2_ans == gt) if p2_ans else False
        p2_iou, p2_rsa = _spatial(br2["valid"], br2["box"], target_box, spec.rsa_valid)

        box_revised    = _boxes_differ(br1["box"], br2["box"])
        answer_flipped = (p1_ans is not None and p2_ans is not None
                          and p1_ans != p2_ans)
        recovered      = (not p1_correct) and p2_correct

        print(f"  [P2] ans={p2_ans}  correct={p2_correct}"
              f"  revised={box_revised}  flipped={answer_flipped}")

        results.append({
            "example_id":                    ex.example_id,
            "caption":                       ex.caption,
            "relation":                      ex.relation,
            "subj":                          ex.subj,
            "obj":                           ex.obj,
            "ground_truth_answer":           gt,
            "target_box":                    target_box,
            "pass1_raw_output":              p1_raw,
            "pass1_parsed_json":             pj1,
            "pass1_reasoning":               parse_reasoning(pj1 or p1_raw, key="reasoning"),
            "pass1_box":                     br1["box"],
            "pass1_box_valid":               br1["valid"],
            "pass1_box_invalid_reason":      br1["invalid_reason"],
            "pass1_answer":                  p1_ans,
            "pass1_answer_correct":          p1_correct,
            "pass1_iou":                     p1_iou,
            "pass1_rsa":                     p1_rsa,
            "pass2_raw_output":              p2_raw,
            "pass2_parsed_json":             pj2,
            "pass2_verification_reasoning":  parse_reasoning(pj2 or p2_raw, key="verification_reasoning"),
            "pass2_box":                     br2["box"],
            "pass2_box_valid":               br2["valid"],
            "pass2_box_invalid_reason":      br2["invalid_reason"],
            "pass2_answer":                  p2_ans,
            "pass2_answer_correct":          p2_correct,
            "pass2_iou":                     p2_iou,
            "pass2_rsa":                     p2_rsa,
            "box_revised":                   box_revised,
            "answer_flipped":                answer_flipped,
            "recovered":                     recovered,
            **meta,
        })

    return results


def _run_multistage(examples, images, spec, model, config) -> list[dict]:
    p1_tmpl    = load_prompt_template("multi_stage_pass1")
    verif_tmpl = load_prompt_template("multi_stage_verification")
    reask_tmpl = load_prompt_template("multi_stage_reask_answer")
    meta       = _meta(config, spec)
    results    = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.example_id}")
        img        = images[ex.example_id]
        gt         = "true" if ex.label else "false"
        target_box = ex.target_box_normalized()

        # Stage 1: initial grounding
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

        # Stage 2: BQS (provisional, stability=1.0 placeholder)
        bqs_prov = compute_bqs(
            box_valid=br1["valid"], box=br1["box"], target_box=target_box,
            reasoning=s1["initial_reasoning"], subj=ex.subj, obj=ex.obj,
            stability_score=1.0,
        )
        triggered = not br1["valid"]

        # Stages 4 & 5: correction path
        if triggered:
            try:
                v_raw = model.generate_response(
                    img,
                    format_prompt_multi(verif_tmpl, caption=ex.caption,
                                        initial_box=_fmt_box(br1["box"])),
                )
            except Exception as exc:
                print(f"  [S4] ERROR: {exc}"); v_raw = ""

            pj4   = extract_json_object(v_raw)
            br4   = parse_box(pj4 or v_raw, mode="visual_cot")
            s4    = {
                "verification_raw_output":    v_raw,
                "verification_parsed_json":   pj4,
                "verification_reasoning":     parse_reasoning(pj4 or v_raw, key="verification_reasoning"),
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
        rev_box        = s4["revised_box"]
        box_revised    = _boxes_differ(s1["initial_box"], rev_box) if triggered else False
        answer_flipped = (s1["initial_answer"] is not None and final_answer is not None
                          and s1["initial_answer"] != final_answer)

        print(f"  [FN] final={final_answer}  correct={final_correct}"
              f"  revised={box_revised}  flipped={answer_flipped}")

        results.append({
            "example_id":          ex.example_id,
            "caption":             ex.caption,
            "relation":            ex.relation,
            "subj":                ex.subj,
            "obj":                 ex.obj,
            "ground_truth_answer": gt,
            "target_box":          target_box,
            **s1,
            "initial_answer_correct":  init_correct,
            "initial_iou":             init_iou,
            "initial_rsa":             init_rsa,
            "bqs":                     bqs_final["bqs"],
            "format_score":            bqs_final["format_score"],
            "overlap_score":           bqs_final["overlap_score"],
            "mention_score":           bqs_final["mention_score"],
            "stability_score":         bqs_final["stability_score"],
            "correction_triggered":    triggered,
            **s4,
            "revised_iou":             rev_iou,
            "revised_rsa":             rev_rsa,
            **reask_fields,
            "final_answer":            final_answer,
            "final_answer_correct":    final_correct,
            "box_revised":             box_revised,
            "answer_flipped":          answer_flipped,
            "recovered":               (not init_correct) and final_correct,
            **meta,
        })

    return results


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_RUNNERS = {
    "textual":      _run_textual,
    "visual":       _run_visual,
    "verification": _run_verification,
    "multistage":   _run_multistage,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval(config: EvalConfig) -> list[dict]:
    """
    Execute one evaluation cell defined by `config`.

    Loads the matched subset, loads the model, runs the method, saves results,
    and returns the per-example result list.
    """
    print(f"\n{'='*60}")
    print(f"  model={config.model_name}  method={config.method}"
          f"  condition={config.condition}  n={config.n_examples}")
    print(f"{'='*60}")

    # Resolve corruption spec
    spec = _SPEC_MAP.get(config.condition, _CLEAN_SPEC)

    # Load data
    print("\nLoading VSR examples...")
    examples = _load_examples(config)
    print(f"  {len(examples)} examples ready.")

    print("\nLoading images...")
    images = _load_images(examples, spec)

    print("\nLoading model...")
    model = _load_model(config)

    # Run
    runner  = _RUNNERS[config.method]
    results = runner(examples, images, spec, model, config)

    # Save
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    save_results(results, config.output_path)

    n = len(results)
    if config.method == "textual":
        correct = sum(1 for r in results if r.get("answer_correct"))
    elif config.method == "visual":
        correct = sum(1 for r in results if r.get("answer_correct"))
    elif config.method == "verification":
        correct = sum(1 for r in results if r.get("pass2_answer_correct"))
    else:
        correct = sum(1 for r in results if r.get("final_answer_correct"))
    print(f"\n  FAA = {correct/n:.3f} ({correct}/{n})")
    print(f"  Saved → {config.output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run one (model, method, condition) evaluation cell.")
    p.add_argument("--model",     default="qwen", choices=["qwen"],
                   help="Model to use (default: qwen)")
    p.add_argument("--method",    required=True,
                   choices=["textual", "visual", "verification", "multistage"])
    p.add_argument("--condition", required=True,
                   help="'clean' or a corruption display name (e.g. blur-1, rot-45)")
    p.add_argument("--subset",    default=None,
                   help="Path to saved subset JSON from eval_config.py (recommended)")
    p.add_argument("--n",         type=int, default=100,
                   help="Number of examples if no subset file given (default: 100)")
    p.add_argument("--seed",      type=int, default=42,
                   help="Subset seed if no subset file given (default: 42)")
    p.add_argument("--out-dir",   default="results/raw",
                   help="Output directory for raw result files (default: results/raw)")
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    cfg = EvalConfig(
        model_name      = args.model,
        method          = args.method,
        condition       = args.condition,
        n_examples      = args.n,
        subset_seed     = args.seed,
        subset_file     = args.subset,
        raw_results_dir = args.out_dir,
        max_new_tokens  = args.max_tokens,
    )

    run_eval(cfg)


if __name__ == "__main__":
    main()
