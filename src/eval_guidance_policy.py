"""
Evaluation of the trained visual guidance policy on the VSR test split.

For each test example:
  1. Predict guidance box (SFT or RL checkpoint)
  2. Compute IoU, RSA@0.5, RSA@0.25 against target relation box
  3. Crop image using predicted box
  4. Ask VLM to answer using one or more modes

Answer modes:
  full      : full image only
  crop      : predicted crop only
  full+crop : full image then crop (two-turn or context prompt)
  oracle    : ground-truth crop (upper bound)

Also computes baselines:
  random    : uniform random box
  full_image: full image box [0,0,1,1]
  prompted  : box from prompted Visual CoT run (if results file provided)

CLI:
    python src/eval_guidance_policy.py \\
        --checkpoint ../checkpoints/guidance_sft/best.pt \\
        --data data/vsr_guidance.json \\
        --model qwen \\
        --modes full crop \\
        --output-dir results/guidance_raw

Outputs:
    results/guidance_raw/{checkpoint_name}_{mode}.json
    results/guidance_aggregated/guidance_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crop_utils import safe_crop
from guidance_dataset import VSRGuidanceDataset, _make_splits
from guidance_model import GuidancePolicy, raw_to_box
from metrics import box_area, rsa_at_threshold
from utils import format_prompt, load_prompt_template

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

VLM_PROMPT_TEMPLATE = (
    "Is the following statement true or false about the image?\n"
    'Statement: "{caption}"\n'
    'Respond with JSON only: {"reasoning": "...", "answer": "true" or "false"}'
)

# Prompt for full+crop mode: model sees full image then the highlighted region
VLM_PROMPT_FULL_CROP = (
    "You are given two images: the full scene and a cropped region of interest.\n"
    "Use both to answer whether the following statement is true or false.\n"
    'Statement: "{caption}"\n'
    'Respond with JSON only: {"reasoning": "...", "answer": "true" or "false"}'
)


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-7)


def _random_box(seed: int) -> list:
    rng = random.Random(seed)
    x1, x2 = sorted([rng.random(), rng.random()])
    y1, y2 = sorted([rng.random(), rng.random()])
    return [x1, y1, max(x1 + 0.02, x2), max(y1 + 0.02, y2)]


# ---------------------------------------------------------------------------
# VLM answering
# ---------------------------------------------------------------------------

def load_vlm(model_name: str):
    if model_name == "qwen":
        from model_interface import QwenVLLocalInterface
        return QwenVLLocalInterface()
    elif model_name == "llava":
        raise NotImplementedError("LLaVA interface not yet implemented in model_interface.py")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def ask_vlm(vlm, image: Image.Image, caption: str) -> tuple[Optional[str], str]:
    """Query VLM with a single image."""
    prompt = VLM_PROMPT_TEMPLATE.replace("{caption}", caption)
    try:
        raw = vlm.generate_response(image, prompt)
        from parse_outputs import extract_json_object, parse_answer
        answer = parse_answer(extract_json_object(raw))
        return answer, raw
    except Exception as e:
        return None, str(e)


def ask_vlm_multi(vlm, full_image: Image.Image, crop: Image.Image, caption: str) -> tuple[Optional[str], str]:
    """Query VLM with full image + crop side by side."""
    prompt = VLM_PROMPT_FULL_CROP.replace("{caption}", caption)
    try:
        raw = vlm.generate_response_multi([full_image, crop], prompt)
        from parse_outputs import extract_json_object, parse_answer
        answer = parse_answer(extract_json_object(raw))
        return answer, raw
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_policy(
    checkpoint: Optional[str],
    data_cache: str,
    model_name: str,
    modes: list[str],
    output_dir: str,
    baselines: list[str],
    split: str,
    seed: int,
    max_examples: Optional[int],
    save_viz: bool,
    viz_dir: Optional[str],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    cache_abs = data_cache if os.path.isabs(data_cache) else os.path.join(_ROOT, data_cache)
    all_ds = VSRGuidanceDataset.from_cache(cache_abs, split="all", seed=seed)
    splits = _make_splits(all_ds.examples, seed)
    test_ds = VSRGuidanceDataset(splits[split])

    if max_examples:
        test_ds = VSRGuidanceDataset(test_ds.examples[:max_examples])
    print(f"Evaluating on {len(test_ds)} {split} examples.")

    # Load guidance policy
    policy = None
    if checkpoint:
        print(f"Loading policy checkpoint: {checkpoint}")
        policy = GuidancePolicy.load_checkpoint(checkpoint, device=device)
        policy.eval()
        processor = policy.processor

    # Load VLM (if any answer modes requested)
    run_vlm = any(m != "grounding_only" for m in modes) and model_name
    vlm = load_vlm(model_name) if run_vlm else None

    results_by_method: dict[str, list[dict]] = {m: [] for m in modes}
    for bsl in baselines:
        results_by_method[f"baseline_{bsl}"] = []

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        image       = item["image"]
        caption     = item["caption"]
        gt_answer   = item["answer"]
        target_box  = item["target_box"]
        example_id  = item["example_id"]

        if image is None:
            print(f"  Skipping {example_id}: image not available")
            continue

        # --- Predict guidance box ---
        pred_box: Optional[list] = None
        if policy is not None:
            inputs = processor(
                text=[caption], images=[image],
                return_tensors="pt", padding=True, truncation=True, max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                boxes = policy(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
            pred_box = boxes[0].cpu().tolist()

        iou_val  = _iou(pred_box, target_box) if pred_box else None
        rsa_50   = rsa_at_threshold(pred_box, target_box, 0.5)
        rsa_25   = rsa_at_threshold(pred_box, target_box, 0.25)
        pred_area = box_area(pred_box) if pred_box else None

        base_record = {
            "example_id":   example_id,
            "caption":      caption,
            "ground_truth": gt_answer,
            "target_box":   target_box,
            "predicted_box": pred_box,
            "iou":          iou_val,
            "rsa_50":       rsa_50,
            "rsa_25":       rsa_25,
            "pred_area":    pred_area,
        }

        # --- VLM answering for each mode ---
        for mode in modes:
            if vlm is None:
                vlm_answer, raw_out = None, ""
            elif mode == "full":
                vlm_answer, raw_out = ask_vlm(vlm, image, caption)
            elif mode == "crop":
                crop = safe_crop(image, pred_box) if pred_box else image
                vlm_answer, raw_out = ask_vlm(vlm, crop, caption)
            elif mode == "oracle":
                vlm_answer, raw_out = ask_vlm(vlm, safe_crop(image, target_box), caption)
            elif mode == "full_crop":
                crop = safe_crop(image, pred_box) if pred_box else image
                vlm_answer, raw_out = ask_vlm_multi(vlm, image, crop, caption)
            else:
                vlm_answer, raw_out = ask_vlm(vlm, image, caption)

            correct = (vlm_answer == gt_answer) if vlm_answer else None

            record = {
                **base_record,
                "answer_mode":     mode,
                "vlm_answer":      vlm_answer,
                "answer_correct":  correct,
                "vlm_raw":         raw_out,
            }
            results_by_method[mode].append(record)

        # --- Baselines ---
        for bsl in baselines:
            if bsl == "random":
                bsl_box = _random_box(seed=abs(hash(example_id)) % 100000)
            elif bsl == "full_image":
                bsl_box = [0.0, 0.0, 1.0, 1.0]
            else:
                bsl_box = pred_box

            bsl_iou  = _iou(bsl_box, target_box) if bsl_box else None
            bsl_rsa5 = rsa_at_threshold(bsl_box, target_box, 0.5)
            bsl_rsa25 = rsa_at_threshold(bsl_box, target_box, 0.25)

            if vlm:
                bsl_image  = safe_crop(image, bsl_box) if bsl_box else image
                bsl_answer, bsl_raw = ask_vlm(vlm, bsl_image, caption)
            else:
                bsl_answer, bsl_raw = None, ""

            results_by_method[f"baseline_{bsl}"].append({
                "example_id":   example_id,
                "caption":      caption,
                "ground_truth": gt_answer,
                "target_box":   target_box,
                "predicted_box": bsl_box,
                "iou":          bsl_iou,
                "rsa_50":       bsl_rsa5,
                "rsa_25":       bsl_rsa25,
                "pred_area":    box_area(bsl_box) if bsl_box else None,
                "vlm_answer":   bsl_answer,
                "answer_correct": (bsl_answer == gt_answer) if bsl_answer else None,
            })

        # Visualization
        if save_viz and pred_box and viz_dir:
            _save_viz(image, pred_box, target_box, example_id, viz_dir)

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(test_ds)} examples done")

    # Save per-method results
    ckpt_name = os.path.splitext(os.path.basename(checkpoint or "guidance"))[0]
    for method, records in results_by_method.items():
        out_path = os.path.join(output_dir, f"{ckpt_name}_{method}.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} results → {out_path}")

    # Print summary table
    _print_summary(results_by_method)

    # Save CSV summary
    _save_csv_summary(results_by_method, ckpt_name, output_dir)


def _print_summary(results_by_method: dict[str, list[dict]]) -> None:
    header = f"{'Method':<30} {'N':>5} {'mIoU':>8} {'RSA50':>8} {'RSA25':>8} {'Area':>8} {'FAA':>8}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for method, records in results_by_method.items():
        if not records:
            continue
        n = len(records)
        ious    = [r["iou"]   for r in records if r.get("iou")   is not None]
        rsa5    = [r["rsa_50"] for r in records if r.get("rsa_50") is not None]
        rsa25   = [r["rsa_25"] for r in records if r.get("rsa_25") is not None]
        areas   = [r["pred_area"] for r in records if r.get("pred_area") is not None]
        correct = [r["answer_correct"] for r in records if r.get("answer_correct") is not None]

        miou  = sum(ious)  / len(ious)  if ious  else None
        mrsa5 = sum(rsa5)  / len(rsa5)  if rsa5  else None
        mrsa25 = sum(rsa25) / len(rsa25) if rsa25 else None
        marea = sum(areas) / len(areas) if areas else None
        faa   = sum(correct) / len(correct) if correct else None

        def _f(v): return f"{v:.4f}" if v is not None else "  ---"
        print(f"{method:<30} {n:>5} {_f(miou):>8} {_f(mrsa5):>8} {_f(mrsa25):>8} {_f(marea):>8} {_f(faa):>8}")
    print("-" * len(header))


def _save_csv_summary(
    results_by_method: dict[str, list[dict]],
    ckpt_name: str,
    output_dir: str,
) -> None:
    import csv
    agg_dir = os.path.join(os.path.dirname(output_dir), "guidance_aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    path = os.path.join(agg_dir, "guidance_summary.csv")

    rows = []
    for method, records in results_by_method.items():
        if not records:
            continue
        n = len(records)
        ious  = [r["iou"]    for r in records if r.get("iou")    is not None]
        rsa5  = [r["rsa_50"] for r in records if r.get("rsa_50") is not None]
        rsa25 = [r["rsa_25"] for r in records if r.get("rsa_25") is not None]
        areas = [r["pred_area"] for r in records if r.get("pred_area") is not None]
        correct = [r["answer_correct"] for r in records if r.get("answer_correct") is not None]

        rows.append({
            "checkpoint": ckpt_name,
            "method":     method,
            "n":          n,
            "mean_iou":   sum(ious)  / len(ious)   if ious  else "",
            "rsa_50":     sum(rsa5)  / len(rsa5)   if rsa5  else "",
            "rsa_25":     sum(rsa25) / len(rsa25)  if rsa25 else "",
            "mean_area":  sum(areas) / len(areas)  if areas else "",
            "faa":        sum(correct) / len(correct) if correct else "",
        })

    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary CSV → {path}")


def _save_viz(
    image: Image.Image,
    pred_box: list,
    target_box: list,
    example_id: str,
    viz_dir: str,
) -> None:
    os.makedirs(viz_dir, exist_ok=True)
    W, H = image.size
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    def _pixel(box):
        return [int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)]

    draw.rectangle(_pixel(target_box), outline=(0, 200, 0), width=3)   # green = GT
    draw.rectangle(_pixel(pred_box),   outline=(255, 0, 0), width=2)   # red   = pred
    img.save(os.path.join(viz_dir, f"{example_id}.jpg"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate visual guidance policy on VSR.")
    p.add_argument("--checkpoint",    default=None,
                   help="Path to guidance policy checkpoint (.pt). If omitted, runs baselines only.")
    p.add_argument("--data",          default="data/vsr_guidance.json")
    p.add_argument("--model",         default="qwen",  choices=["qwen", "llava", "none"])
    p.add_argument("--modes",         nargs="+", default=["full", "crop", "full_crop"],
                   choices=["full", "crop", "oracle", "full_crop"],
                   help="VLM answer modes to evaluate")
    p.add_argument("--baselines",     nargs="+", default=["random", "full_image"],
                   choices=["random", "full_image"])
    p.add_argument("--split",         default="test", choices=["train", "val", "test"])
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max-examples",  type=int, default=None)
    p.add_argument("--save-viz",      action="store_true",
                   help="Save visualization images (GT + pred boxes)")
    p.add_argument("--viz-dir",       default="results/guidance_viz")
    p.add_argument("--output-dir",    default="results/guidance_raw")
    args = p.parse_args()

    if args.model == "none":
        args.model = None
        args.modes = []

    eval_policy(
        checkpoint    = args.checkpoint,
        data_cache    = args.data,
        model_name    = args.model,
        modes         = args.modes,
        output_dir    = args.output_dir,
        baselines     = args.baselines,
        split         = args.split,
        seed          = args.seed,
        max_examples  = args.max_examples,
        save_viz      = args.save_viz,
        viz_dir       = args.viz_dir if args.save_viz else None,
    )


if __name__ == "__main__":
    main()
